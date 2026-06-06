#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// config

constexpr int    NUM_NICS   = 4;
constexpr int    NUM_GPUS   = 8;
constexpr int    IB_PORT    = 1;
constexpr int    GID_INDEX  = 0;    // 0 for IB; 3 for RoCEv2
constexpr int    TCP_PORT   = 12345;
constexpr int    CHUNK_NUM  = 4;
constexpr size_t CHUNK_SIZE = 1u << 20;   // 1 MB
constexpr size_t TOTAL_SIZE = CHUNK_NUM * CHUNK_SIZE;
struct ConnInfo  { uint32_t qpn; uint16_t lid; uint8_t gid[16]; };
struct ChunkInfo { uint64_t addr; uint32_t rkey; };

#define CUDA_CHECK(cmd)                                                   \
  do {                                                                    \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA %s:%d  %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(e));                                      \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

// Open CX7 NIC 
static ibv_context* open_nic() {
    int n = 0;
    ibv_device** devs = ibv_get_device_list(&n);
    if (!devs || n == 0) { fprintf(stderr, "no IB devices\n"); exit(1); }

    // Filter CX7 devices
    ibv_context* ctx = nullptr;
    for (int i = 0; i < n && !ctx; i++) {
        const char* name = ibv_get_device_name(devs[i]);
        if (strncmp(name, "mlx5_", 5) == 0) {
            ctx = ibv_open_device(devs[i]);
        }
    }
    ibv_free_device_list(devs);
    if (!ctx) { 
        fprintf(stderr, "cannot open mlx5 device %d\n");
        exit(1); 
        
    }
    return ctx;
}

static ibv_qp* create_rc_qp(ibv_context* ctx, ibv_pd* pd, ibv_cq* cq) {
    ibv_qp_init_attr_ex attr = {};
    attr.comp_mask      = IBV_QP_INIT_ATTR_PD | 
                          IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    attr.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE |
                          IBV_QP_EX_WITH_BIND_MW |
                          IBV_QP_EX_WITH_LOCAL_INV;
    attr.pd             = pd;
    attr.send_cq        = cq;
    attr.recv_cq        = cq;
    attr.qp_type        = IBV_QPT_RC;
    attr.cap            = { .max_send_wr=64, .max_recv_wr=1,
                            .max_send_sge=1, .max_recv_sge=1 };
    ibv_qp* qp = ibv_create_qp_ex(ctx, &attr);
    if (!qp) { perror("ibv_create_qp_ex"); exit(1); }
    return qp;
}

static void connect_qp(ibv_context* ctx, ibv_qp* qp, int fd, bool is_server) {
    ibv_port_attr pa = {};
    ibv_query_port(ctx, IB_PORT, &pa);
    union ibv_gid gid = {};
    ibv_query_gid(ctx, IB_PORT, GID_INDEX, &gid);

    ConnInfo my = { .qpn=qp->qp_num, .lid=pa.lid };
    ConnInfo peer = {};
    memcpy(my.gid, &gid, 16);

    if (is_server) { 
        send(fd, &my, sizeof(my), 0);
        recv(fd, &peer, sizeof(peer), MSG_WAITALL);
    }
    else { recv(fd, &peer, sizeof(peer), MSG_WAITALL); 
         send(fd, &my, sizeof(my), 0); 
    }

    // INIT
    ibv_qp_attr a = {};
    a.qp_state        = IBV_QPS_INIT;
    a.port_num        = IB_PORT;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    ibv_modify_qp(qp, &a, IBV_QP_STATE|IBV_QP_PKEY_INDEX|
                           IBV_QP_PORT|IBV_QP_ACCESS_FLAGS);

    // RTR
    memset(&a, 0, sizeof(a));
    a.qp_state              = IBV_QPS_RTR;
    a.path_mtu              = IBV_MTU_4096;
    a.dest_qp_num           = peer.qpn;
    a.rq_psn                = 0;
    a.max_dest_rd_atomic    = 1;
    a.min_rnr_timer         = 12;
    a.ah_attr.port_num      = IB_PORT;
    a.ah_attr.dlid          = peer.lid;
    a.ah_attr.is_global     = 1;          // needed even on IB for GRH
    memcpy(&a.ah_attr.grh.dgid, peer.gid, 16);
    a.ah_attr.grh.sgid_index = GID_INDEX;
    a.ah_attr.grh.hop_limit  = 64;
    ibv_modify_qp(qp, &a, IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|
                  IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|
                  IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER);

    // RTS
    memset(&a, 0, sizeof(a));
    a.qp_state      = IBV_QPS_RTS;
    a.sq_psn        = 0;
    a.timeout       = 14;
    a.retry_cnt     = 7;
    a.rnr_retry     = 7;
    a.max_rd_atomic = 1;
    ibv_modify_qp(qp, &a, IBV_QP_STATE|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                  IBV_QP_RNR_RETRY|IBV_QP_SQ_PSN|IBV_QP_MAX_QP_RD_ATOMIC);
}

static ibv_wc poll_one(ibv_cq* cq) {
    ibv_wc wc = {};
    while (ibv_poll_cq(cq, 1, &wc) == 0);
    return wc;
}

// Rebind MW[i] with given flags; flags=0 revokes. Returns new active rkey.
static uint32_t bind_mw(ibv_qp* qp, ibv_cq* cq,
                         ibv_mw* mw, ibv_mr* mr, void* buf, int i, int flags) {
    ibv_qp_ex* qpx = ibv_qp_to_qp_ex(qp);
    ibv_wr_start(qpx);
    qpx->wr_id    = (uint64_t)i;
    qpx->wr_flags = IBV_SEND_SIGNALED;

    ibv_mw_bind_info bi = {
        .mr              = mr,
        .addr            = (uint64_t)buf + (uint64_t)i * CHUNK_SIZE,
        .length          = CHUNK_SIZE,
        .mw_access_flags = (uint32_t)flags,
    };
    ibv_wr_bind_mw(qpx, mw, mw->rkey, &bi);
    ibv_wr_complete(qpx);

    ibv_wc wc = poll_one(cq);
    if (wc.status != IBV_WC_SUCCESS)
        fprintf(stderr, "bind_mw[%d]: %s\n", i, ibv_wc_status_str(wc.status));

    return mw->rkey;  // driver updates mw->rkey after successful bind
}

static void invalidate_mw(ibv_qp* qp, ibv_cq* cq, ibv_mw* mw, int i) {
    ibv_qp_ex* qpx = ibv_qp_to_qp_ex(qp);
    ibv_wr_start(qpx);
    qpx->wr_id    = (uint64_t)i;
    qpx->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_local_inv(qpx, mw->rkey);
    ibv_wr_complete(qpx);
    ibv_wc wc = poll_one(cq);
    if (wc.status != IBV_WC_SUCCESS)
        fprintf(stderr, "local_inv[%d]: %s\n", i, ibv_wc_status_str(wc.status));
}

// receiver
static void run_server(int gpu_id) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    void* buf = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, TOTAL_SIZE));
    CUDA_CHECK(cudaMemset(buf, 0, TOTAL_SIZE));

    ibv_context* ctx = open_nic();
    ibv_pd*  pd  = ibv_alloc_pd(ctx);
    ibv_cq*  cq  = ibv_create_cq(ctx, 128, nullptr, nullptr, 0);
    ibv_qp*  qp  = create_rc_qp(ctx, pd, cq);
    if (!pd || !cq) { perror("pd/cq"); exit(1); }

    // GPUDirect RDMA: register GPU memory directly
    ibv_mr* mr = ibv_reg_mr(pd, buf, TOTAL_SIZE,
                             IBV_ACCESS_LOCAL_WRITE | 
                             IBV_ACCESS_REMOTE_WRITE |
                             IBV_ACCESS_MW_BIND);
    if (!mr) { perror("ibv_reg_mr (GPU mem)"); exit(1); }

    ibv_mw* mw[CHUNK_NUM];
    for (int i = 0; i < CHUNK_NUM; i++) {
        mw[i] = ibv_alloc_mw(pd, IBV_MW_TYPE_2);
        if (!mw[i]) { perror("ibv_alloc_mw"); exit(1); }
    }

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1; setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    sockaddr_in sa = { .sin_family=AF_INET, .sin_port=htons(TCP_PORT),
                       .sin_addr={INADDR_ANY} };
    bind(lfd, (sockaddr*)&sa, sizeof(sa));
    listen(lfd, 1);
    printf("[Server] GPU=%d  NIC=mlx5_%d  listening :%d\n",
           gpu_id, gpu_id % NUM_NICS, TCP_PORT);
    int fd = accept(lfd, nullptr, nullptr);
    close(lfd);

    connect_qp(ctx, qp, fd, true);
    printf("[Server] QP connected\n\n");

    for (int i = 0; i < CHUNK_NUM; i++) {
        // grant
        uint32_t rkey = bind_mw(qp, cq, mw[i], mr, buf, i, IBV_ACCESS_REMOTE_WRITE);
        printf("[Server] chunk%d  GRANTED  rkey=0x%08x\n", i, rkey);

        ChunkInfo ci = { .addr=(uint64_t)buf + (uint64_t)i*CHUNK_SIZE, .rkey=rkey };
        send(fd, &ci, sizeof(ci), 0);

        uint8_t sig; recv(fd, &sig, 1, MSG_WAITALL);  // wait write-done

        // revoke: rebind with access_flags=0 → old rkey invalidated
        invalidate_mw(qp, cq, mw[i], i);
        printf("[Server] chunk%d  REVOKED\n", i);

        send(fd, &sig, 1, 0);  // notify client
    }

    close(fd);
    printf("\n[Server] done\n");

    for (int i = 0; i < CHUNK_NUM; i++) ibv_dealloc_mw(mw[i]);
    ibv_dereg_mr(mr);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    CUDA_CHECK(cudaFree(buf));
}

// sender
static void run_client(const char* ip, int gpu_id) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    void* buf = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, TOTAL_SIZE));
    CUDA_CHECK(cudaMemset(buf, 0xAB, TOTAL_SIZE));

    ibv_context* ctx = open_nic();
    ibv_pd*  pd  = ibv_alloc_pd(ctx);
    ibv_cq*  cq  = ibv_create_cq(ctx, 128, nullptr, nullptr, 0);
    ibv_qp*  qp  = create_rc_qp(ctx, pd, cq);
    if (!pd || !cq) { perror("pd/cq"); exit(1); }

    // GPUDirect RDMA: local send buffer from GPU memory
    ibv_mr* mr = ibv_reg_mr(pd, buf, TOTAL_SIZE, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) { perror("ibv_reg_mr (GPU mem)"); exit(1); }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in sa = { .sin_family=AF_INET, .sin_port=htons(TCP_PORT) };
    inet_pton(AF_INET, ip, &sa.sin_addr);
    while (connect(fd, (sockaddr*)&sa, sizeof(sa)) != 0) usleep(100000);

    connect_qp(ctx, qp, fd, false);
    printf("[Client] GPU=%d  NIC=mlx5_%d  QP connected\n\n",
           gpu_id, gpu_id % NUM_NICS);

    for (int i = 0; i < CHUNK_NUM; i++) {
        ChunkInfo ci;
        recv(fd, &ci, sizeof(ci), MSG_WAITALL);
        printf("[Client] chunk%d  rkey=0x%08x  addr=0x%lx\n", i, ci.rkey, ci.addr);

        ibv_sge sge = {
            .addr   = (uint64_t)buf + (uint64_t)i * CHUNK_SIZE,
            .length = (uint32_t)CHUNK_SIZE,
            .lkey   = mr->lkey,
        };
        ibv_send_wr wr = {};
        wr.wr_id      = (uint64_t)i;
        wr.opcode     = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma    = { .remote_addr=ci.addr, .rkey=ci.rkey };
        wr.sg_list    = &sge;
        wr.num_sge    = 1;

        ibv_send_wr* bad;
        ibv_post_send(qp, &wr, &bad);
        ibv_wc wc = poll_one(cq);
        printf("[Client] chunk%d  write (valid rkey):   %s\n",
               i, ibv_wc_status_str(wc.status));

        uint8_t sig = 1; send(fd, &sig, 1, 0);      // write done
        recv(fd, &sig, 1, MSG_WAITALL);              // wait revocation

        // same wr, same old rkey → expect IBV_WC_REM_ACCESS_ERR
        ibv_post_send(qp, &wr, &bad);
        wc = poll_one(cq);
        bool blocked = (wc.status == IBV_WC_REM_ACCESS_ERR ||
                wc.status == IBV_WC_RETRY_EXC_ERR);
        printf("[Client] chunk%d  write (revoked rkey): %s  %s\n\n",
               i, ibv_wc_status_str(wc.status), blocked ? "✓ BLOCKED" : "✗ UNEXPECTED");

        // RC QP enters ERR state after remote access error — demo ends
        if (blocked) break;
    }

    close(fd);
    printf("[Client] done\n");

    ibv_dereg_mr(mr);
    ibv_destroy_qp(qp);
    ibv_destroy_cq(cq);
    ibv_dealloc_pd(pd);
    ibv_close_device(ctx);
    CUDA_CHECK(cudaFree(buf));
}

int main(int argc, char* argv[]) {
    // SERVER_GPU / CLIENT_GPU env vars override default (GPU 0)
    int gpu_id = 0;
    const char* env = getenv(argc == 1 ? "SERVER_GPU" : "CLIENT_GPU");
    if (env) gpu_id = atoi(env);

    if (argc == 1)       run_server(gpu_id);
    else if (argc >= 2)  run_client(argv[1], argc >= 3 ? atoi(argv[2]) : gpu_id);
    return 0;
}