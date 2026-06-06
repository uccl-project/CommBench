/*
 * Demonstrates ibv_get_async_event() — the mechanism for receiving
 * device-level asynchronous error and state-change notifications.
 *
 * Both async events and CQ completion events share the same pattern:
 *
 *   Async events (device/QP/CQ errors):
 *     select(ctx->async_fd) → ibv_get_async_event() → ibv_ack_async_event()
 *
 *   CQ completion events (separate example):
 *     select(cc->fd) → ibv_get_cq_event() → ibv_ack_cq_events()
 *
 * Key concepts:
 *   1. ctx->async_fd          — file descriptor for async event delivery
 *   2. fcntl(O_NONBLOCK)      — enable non-blocking mode for select() usage
 *   3. ibv_get_async_event()  — retrieve one event from the fd
 *   4. ibv_ack_async_event()  — MUST ack: until acked, ibv_destroy_cq() /
 *                               ibv_destroy_qp() will block indefinitely
 *
 * Setup: two RDMA NICs (NIC 0 = sender, NIC 1 = receiver).
 *   NIC 0 owns the tiny CQ, the sending QP (qp0), and the source buffer.
 *   NIC 1 owns the receiving QP (qp1) and the destination buffer.
 *
 * Trigger: CQ overflow (IBV_EVENT_CQ_ERR)
 *   A tiny CQ is created on NIC 0.  Posting more signaled RDMA writes from
 *   qp0 to qp1's buffer than the CQ can hold causes it to overflow, which
 *   delivers IBV_EVENT_CQ_ERR on NIC 0's ctx->async_fd.
 *
 * Benchmark: measures async event detection latency (time from first
 * ibv_post_send to ibv_get_async_event() returning) across multiple trials.
 */

 #include <infiniband/verbs.h>
 #include <sys/select.h>
 #include <sys/time.h>
 #include <fcntl.h>
 #include <unistd.h>
 
 #include <cstdlib>
 #include <cstring>
 #include <iostream>
 #include <stdexcept>
 #include <string>
 #include <vector>
 
 // ── Constants ────────────────────────────────────────────────────────────────
 
 static constexpr int PORT_NUM    = 1;
 // MAX_SEND_WR must be >= (actual CQ entries + 2) after driver rounding.
 // mlx5 rounds TINY_CQ_SIZE=1 up significantly (e.g. to 63 entries);
 // 256 is a safe upper bound for all known mlx5 generations.
 static constexpr int MAX_SEND_WR = 256;
 static constexpr int MAX_RECV_WR = 16;
 static constexpr int MAX_SGE     = 1;
 // Request a tiny CQ; mlx5 rounds up (e.g. 1 → 63 actual entries).
 // After creation we read cq->cqe for the real capacity and post cqe+2 writes.
 static constexpr int TINY_CQ_SIZE = 1;
 
 #define CHECK(cond, msg) \
     do { if (!(cond)) throw std::runtime_error(std::string(msg)); } while (0)
 #define CHECK_PTR(ptr, msg) \
     do { if (!(ptr)) { perror(msg); throw std::runtime_error(std::string(msg)); } } while (0)
 
 // ── AsyncEventMonitor ─────────────────────────────────────────────────────────
 //
 // Wraps ctx->async_fd with:
 //   - non-blocking fd setup (so select() can be used with a timeout)
 //   - select()-based waiting with timeout
 //   - retrieval via ibv_get_async_event()
 //   - mandatory ack via ibv_ack_async_event()
 //
 
 class AsyncEventMonitor {
 public:
     explicit AsyncEventMonitor(struct ibv_context* ctx) : ctx_(ctx) {}
 
     AsyncEventMonitor(const AsyncEventMonitor&)            = delete;
     AsyncEventMonitor& operator=(const AsyncEventMonitor&) = delete;
 
     /**
      * Set ctx->async_fd to non-blocking mode.
      *
      * Uses fcntl(F_GETFL) then fcntl(F_SETFL, flags | O_NONBLOCK).
      * Required so that select() with a timeout can be used safely.
      *
      * @return true on success
      */
     bool setNonBlocking() {
         // NOTE: safe only from a single thread.  In a multi-threaded scenario
         // a race between select() returning readable and the subsequent
         // ibv_get_async_event() call could yield EAGAIN on the O_NONBLOCK fd.
         // TODO: set O_NONBLOCK on ctx_->async_fd using fcntl(F_GETFL) / fcntl(F_SETFL)
         // Return true on success, false on error.
     }
 
     /**
      * Wait for an async event using select() with timeout, then retrieve
      * it via ibv_get_async_event().
      *
      * @param event       Output: the retrieved async event
      * @param timeout_ms  Timeout in milliseconds (0 = non-blocking poll)
      * @return true if an event was received, false on timeout / error
      */
     bool waitEvent(struct ibv_async_event* event, int timeout_ms) {
         // TODO: build fd_set with ctx_->async_fd, call select() with timeout,
         // then call ibv_get_async_event(ctx_, event) if select() > 0.
         // Return false on timeout (select returns 0) or error.
     }
 
     /**
      * Acknowledge an async event.
      *
      * MUST be called once per received event.  Until acked, the reference
      * count on the associated object (CQ / QP / SRQ / device) is held —
      * ibv_destroy_cq() / ibv_destroy_qp() will block indefinitely if any
      * event remains unacked.
      *
      * @param event  The event to acknowledge (as returned by waitEvent)
      */
     void ackEvent(struct ibv_async_event* event) {
         // TODO: call ibv_ack_async_event(event)
     }
 
     /** The raw async event fd, for external select()/poll(). */
     int getAsyncFd() const { return ctx_->async_fd; }
 
 private:
     struct ibv_context* ctx_;
 };
 
 // ── NicInfo ───────────────────────────────────────────────────────────────────
 //
 // Pre-queried information about one RDMA NIC.
 // Filled once by queryNic() at the start of runTest(); cheaply copied
 // into TrialResources and warmUpPath without re-querying the driver.
 //
 
 struct NicInfo {
     struct ibv_context*  ctx;
     struct ibv_pd*       pd;
     int                  gid_index;
     union ibv_gid        gid;
     struct ibv_port_attr port_attr;
 };
 
 static int findGidIndex(struct ibv_context* ctx) {
     const char* env = getenv("RDMA_GID_INDEX");
     if (env) return std::atoi(env);
     struct ibv_port_attr pa;
     if (ibv_query_port(ctx, PORT_NUM, &pa) != 0) return 0;
     for (int i = 0; i < pa.gid_tbl_len && i < 16; i++) {
         union ibv_gid g;
         if (ibv_query_gid(ctx, PORT_NUM, i, &g) != 0) continue;
         bool zero = true;
         for (int j = 0; j < 16; j++) if (g.raw[j]) { zero = false; break; }
         if (zero) continue;
         if (g.raw[10] == 0xff && g.raw[11] == 0xff) return i;
     }
     return 0;
 }
 
 static NicInfo queryNic(struct ibv_context* ctx, struct ibv_pd* pd) {
     NicInfo n;
     n.ctx = ctx;
     n.pd  = pd;
     CHECK(ibv_query_port(ctx, PORT_NUM, &n.port_attr) == 0, "ibv_query_port");
     n.gid_index = findGidIndex(ctx);
     CHECK(ibv_query_gid(ctx, PORT_NUM, n.gid_index, &n.gid) == 0, "ibv_query_gid");
     return n;
 }
 
 // ── connectQP ─────────────────────────────────────────────────────────────────
 //
 // Transition 'qp' from RST → INIT → RTR → RTS.
 //   local_n:  NicInfo for the device that owns 'qp'
 //   remote_n: NicInfo for the remote device (where packets will be sent)
 //
 
 static void connectQP(struct ibv_qp* qp, uint32_t remote_qpn,
                       const NicInfo& local_n, const NicInfo& remote_n) {
     struct ibv_qp_attr a = {};
 
     // INIT
     a.qp_state        = IBV_QPS_INIT;
     a.port_num        = PORT_NUM;
     a.pkey_index      = 0;
     a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                         IBV_ACCESS_REMOTE_WRITE |
                         IBV_ACCESS_REMOTE_READ;
     CHECK(ibv_modify_qp(qp, &a,
         IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
         IBV_QP_ACCESS_FLAGS) == 0, "QP to INIT");
 
     // RTR — AH points to the remote NIC
     memset(&a, 0, sizeof(a));
     a.qp_state           = IBV_QPS_RTR;
     // Use the smaller MTU of the two ports to avoid packet fragmentation
     a.path_mtu           = (local_n.port_attr.active_mtu <= remote_n.port_attr.active_mtu)
                             ? local_n.port_attr.active_mtu
                             : remote_n.port_attr.active_mtu;
     a.dest_qp_num        = remote_qpn;
     a.rq_psn             = 0;
     a.max_dest_rd_atomic = 1;
     a.min_rnr_timer      = 12;
     if (local_n.port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
         // IB: reach the remote port via its LID
         a.ah_attr.is_global     = 0;
         a.ah_attr.dlid          = remote_n.port_attr.lid;
         a.ah_attr.sl            = 0;
         a.ah_attr.src_path_bits = 0;
         a.ah_attr.port_num      = PORT_NUM;
     } else {
         // RoCE: GRH with the remote NIC's GID as destination
         a.ah_attr.is_global          = 1;
         a.ah_attr.port_num           = PORT_NUM;
         a.ah_attr.grh.hop_limit      = 1;
         a.ah_attr.grh.dgid           = remote_n.gid;        // remote NIC's GID
         a.ah_attr.grh.sgid_index     = local_n.gid_index;   // local NIC's GID index
     }
     CHECK(ibv_modify_qp(qp, &a,
         IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
         IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
         IBV_QP_MIN_RNR_TIMER | IBV_QP_AV) == 0, "QP to RTR");
 
     // RTS
     memset(&a, 0, sizeof(a));
     a.qp_state      = IBV_QPS_RTS;
     a.timeout       = 14;
     a.retry_cnt     = 7;
     a.rnr_retry     = 7;
     a.sq_psn        = 0;
     a.max_rd_atomic = 1;
     CHECK(ibv_modify_qp(qp, &a,
         IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
         IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
         IBV_QP_MAX_QP_RD_ATOMIC) == 0, "QP to RTS");
 }
 
 // ── TrialResources ────────────────────────────────────────────────────────────
 //
 // Per-trial setup for dual-NIC RDMA write overflow:
 //   NIC 0 (n0): tiny CQ + send QP (qp0_) + source MR (mr0_)
 //   NIC 1 (n1): normal CQ + recv QP (qp1_) + destination MR (mr1_)
 //
 // Posting cqe+2 signaled writes from qp0_ to mr1_'s buffer fills the tiny
 // CQ on NIC 0, triggering IBV_EVENT_CQ_ERR on NIC 0's ctx->async_fd.
 //
 // The destructor is safe only after all async events from this CQ / QPs
 // have been acked by the caller.
 //
 
 class TrialResources {
 public:
     TrialResources(const NicInfo& n0, const NicInfo& n1) {
         int acc = IBV_ACCESS_LOCAL_WRITE |
                   IBV_ACCESS_REMOTE_WRITE |
                   IBV_ACCESS_REMOTE_READ;
 
         // ── NIC 0: tiny CQ (will overflow), send QP, source buffer ───────
         cq_ = ibv_create_cq(n0.ctx, TINY_CQ_SIZE, nullptr, nullptr, 0);
         CHECK_PTR(cq_, "ibv_create_cq (tiny)");
         actual_cqe_ = cq_->cqe;  // driver-rounded actual capacity
 
         buf0_ = aligned_alloc(4096, 4096);
         CHECK_PTR(buf0_, "aligned_alloc buf0");
         memset(buf0_, 0xAB, 4096);
 
         mr0_ = ibv_reg_mr(n0.pd, buf0_, 4096, acc);
         CHECK_PTR(mr0_, "ibv_reg_mr mr0");
 
         // qp0_: both send_cq and recv_cq point to the tiny cq_ on NIC 0.
         // RDMA_WRITE generates only send completions, so cq_ will overflow.
         qp0_ = makeQP(n0.pd, cq_);
 
         // ── NIC 1: normal CQ (won't overflow), recv QP, destination buffer
         cq1_ = ibv_create_cq(n1.ctx, 64, nullptr, nullptr, 0);
         CHECK_PTR(cq1_, "ibv_create_cq (normal)");
 
         buf1_ = aligned_alloc(4096, 4096);
         CHECK_PTR(buf1_, "aligned_alloc buf1");
         memset(buf1_, 0x00, 4096);
 
         mr1_ = ibv_reg_mr(n1.pd, buf1_, 4096, acc);
         CHECK_PTR(mr1_, "ibv_reg_mr mr1");
 
         // qp1_: receives RDMA writes from NIC 0 (no CQEs for RDMA_WRITE
         // on the receiver side — cq1_ stays empty throughout the test).
         qp1_ = makeQP(n1.pd, cq1_);
 
         // Connect: qp0_(NIC0) sends to qp1_(NIC1), qp1_ accepts from qp0_
         connectQP(qp0_, qp1_->qp_num, n0, n1);
         connectQP(qp1_, qp0_->qp_num, n1, n0);
 
         // Per-trial RC warm-up: post one write and spin-poll until it completes.
         //
         // Each trial creates a fresh QP pair.  A new RC QP's first packet goes
         // through a slow path in the NIC / switch (flow-table population, path
         // resolution) that can add hundreds of microseconds — and varies wildly
         // between trials.  By issuing one dummy write here (before t_trigger)
         // and draining its completion from the tiny CQ, we ensure the hardware
         // path is fully established before timing begins.
         //
         // After this the tiny CQ is empty; postOverflowWrites() will fill it.
         {
             struct ibv_sge sge = {};
             sge.addr   = reinterpret_cast<uint64_t>(buf0_);
             sge.length = 64;
             sge.lkey   = mr0_->lkey;
 
             struct ibv_send_wr wr = {};
             wr.opcode              = IBV_WR_RDMA_WRITE;
             wr.send_flags          = IBV_SEND_SIGNALED;
             wr.sg_list             = &sge;
             wr.num_sge             = 1;
             wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(buf1_);
             wr.wr.rdma.rkey        = mr1_->rkey;
 
             struct ibv_send_wr* bad = nullptr;
             CHECK(ibv_post_send(qp0_, &wr, &bad) == 0, "warm-up ibv_post_send");
 
             // Spin-poll: the completion goes into the tiny CQ; drain it so the
             // CQ is empty when timing starts.
             struct ibv_wc wc = {};
             while (ibv_poll_cq(cq_, 1, &wc) == 0) {}
             CHECK(wc.status == IBV_WC_SUCCESS, "warm-up write returned error status");
         }
     }
 
     ~TrialResources() {
         // Safe to destroy only after all async events (CQ_ERR, QP_FATAL)
         // from these resources have been acked by the caller.
         if (qp0_) ibv_destroy_qp(qp0_);
         if (qp1_) ibv_destroy_qp(qp1_);
         if (mr0_) ibv_dereg_mr(mr0_);
         if (mr1_) ibv_dereg_mr(mr1_);
         if (buf0_) free(buf0_);
         if (buf1_) free(buf1_);
         if (cq_)   ibv_destroy_cq(cq_);   // NIC0 tiny CQ — blocks if events unacked!
         if (cq1_)  ibv_destroy_cq(cq1_);  // NIC1 normal CQ
     }
 
     TrialResources(const TrialResources&)            = delete;
     TrialResources& operator=(const TrialResources&) = delete;
 
     /**
      * Returns the raw cq->cqe value (mlx5 ring-buffer mask).
      * Actual CQ capacity = cqe+1; need cqe+2 writes to guarantee overflow.
      */
     int getActualCqe() const { return actual_cqe_; }
 
     /**
      * Post n signaled RDMA writes from qp0_ (NIC 0) to qp1_'s buffer (NIC 1).
      *
      * Each write generates one send completion in the tiny CQ on NIC 0.
      * Once completions exceed the actual CQ capacity (cq->cqe+1), the CQ
      * overflows and IBV_EVENT_CQ_ERR is delivered to NIC 0's ctx->async_fd.
      *
      * @param n  Number of writes — use getActualCqe()+2 to guarantee overflow.
      */
     void postOverflowWrites(int n) {
         // TODO: loop n times, posting one IBV_WR_RDMA_WRITE with IBV_SEND_SIGNALED
         // from qp0_ (source: buf0_ on NIC 0, lkey=mr0_->lkey, length=64) to
         // qp1_'s buffer on NIC 1 (remote_addr=buf1_, rkey=mr1_->rkey).
         // Break early if ibv_post_send() returns non-zero.
     }
 
 private:
     struct ibv_cq* cq_         = nullptr;  // tiny CQ on NIC 0
     struct ibv_cq* cq1_        = nullptr;  // normal CQ on NIC 1
     struct ibv_qp* qp0_        = nullptr;  // send QP on NIC 0
     struct ibv_qp* qp1_        = nullptr;  // recv QP on NIC 1
     struct ibv_mr* mr0_        = nullptr;  // source MR on NIC 0
     struct ibv_mr* mr1_        = nullptr;  // destination MR on NIC 1
     void*          buf0_       = nullptr;  // source buffer
     void*          buf1_       = nullptr;  // destination buffer
     int            actual_cqe_ = 0;        // real CQ capacity after driver rounding
 
     static struct ibv_qp* makeQP(struct ibv_pd* pd, struct ibv_cq* cq) {
         struct ibv_qp_init_attr attr = {};
         attr.send_cq          = cq;
         attr.recv_cq          = cq;
         attr.qp_type          = IBV_QPT_RC;
         attr.sq_sig_all       = 0;
         attr.cap.max_send_wr  = MAX_SEND_WR;
         attr.cap.max_recv_wr  = MAX_RECV_WR;
         attr.cap.max_send_sge = MAX_SGE;
         attr.cap.max_recv_sge = MAX_SGE;
         struct ibv_qp* qp = ibv_create_qp(pd, &attr);
         CHECK_PTR(qp, "ibv_create_qp");
         return qp;
     }
 };
 
 // ── Path warm-up ─────────────────────────────────────────────────────────────
 //
 // The first RDMA operation across a physical link triggers hardware path
 // initialization (ARP / neighbor-discovery) that can take several seconds.
 // Use the completion channel — not the async fd — to wait for this first
 // operation before starting the timed trials.
 //
 
 static void warmUpPath(const NicInfo& n0, const NicInfo& n1) {
     struct ibv_comp_channel* cc = ibv_create_comp_channel(n0.ctx);
     if (!cc) return;
 
     int fl = fcntl(cc->fd, F_GETFL, 0);
     fcntl(cc->fd, F_SETFL, fl | O_NONBLOCK);
 
     // CQ with comp channel on NIC 0 (sender), plain CQ on NIC 1 (receiver)
     struct ibv_cq* wcq0 = ibv_create_cq(n0.ctx, 64, nullptr, cc, 0);
     struct ibv_cq* wcq1 = ibv_create_cq(n1.ctx, 64, nullptr, nullptr, 0);
     if (!wcq0 || !wcq1) {
         if (wcq0) ibv_destroy_cq(wcq0);
         if (wcq1) ibv_destroy_cq(wcq1);
         ibv_destroy_comp_channel(cc);
         return;
     }
 
     auto makeWuQP = [](struct ibv_pd* pd, struct ibv_cq* cq) {
         struct ibv_qp_init_attr a = {};
         a.send_cq = a.recv_cq = cq;
         a.qp_type = IBV_QPT_RC;
         a.cap.max_send_wr = 16; a.cap.max_recv_wr = 16;
         a.cap.max_send_sge = a.cap.max_recv_sge = 1;
         return ibv_create_qp(pd, &a);
     };
 
     struct ibv_qp* wq0 = makeWuQP(n0.pd, wcq0);  // sender on NIC 0
     struct ibv_qp* wq1 = makeWuQP(n1.pd, wcq1);  // receiver on NIC 1
     if (!wq0 || !wq1) {
         if (wq0) ibv_destroy_qp(wq0);
         if (wq1) ibv_destroy_qp(wq1);
         ibv_destroy_cq(wcq0); ibv_destroy_cq(wcq1);
         ibv_destroy_comp_channel(cc);
         return;
     }
 
     // Connect: wq0(NIC0) → wq1(NIC1), wq1(NIC1) → wq0(NIC0)
     connectQP(wq0, wq1->qp_num, n0, n1);
     connectQP(wq1, wq0->qp_num, n1, n0);
 
     // Register source buffer on NIC 0 and destination buffer on NIC 1
     int acc = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
     char* wb0 = static_cast<char*>(aligned_alloc(4096, 4096));
     char* wb1 = static_cast<char*>(aligned_alloc(4096, 4096));
     if (!wb0 || !wb1) {
         if (wb0) free(wb0);
         if (wb1) free(wb1);
         ibv_destroy_qp(wq0); ibv_destroy_qp(wq1);
         ibv_destroy_cq(wcq0); ibv_destroy_cq(wcq1);
         ibv_destroy_comp_channel(cc);
         return;
     }
     memset(wb0, 0, 4096);
     memset(wb1, 0, 4096);
     struct ibv_mr* wmr0 = ibv_reg_mr(n0.pd, wb0, 4096, acc);
     struct ibv_mr* wmr1 = ibv_reg_mr(n1.pd, wb1, 4096, acc);
 
     if (wmr0 && wmr1) {
         // Arm comp channel, post one RDMA write from NIC 0 to NIC 1's buffer
         ibv_req_notify_cq(wcq0, 0);
         struct ibv_sge s = {(uint64_t)wb0, 64, wmr0->lkey};
         struct ibv_send_wr w = {};
         w.opcode = IBV_WR_RDMA_WRITE; w.send_flags = IBV_SEND_SIGNALED;
         w.sg_list = &s; w.num_sge = 1;
         w.wr.rdma.remote_addr = (uint64_t)wb1;
         w.wr.rdma.rkey = wmr1->rkey;
         struct ibv_send_wr* bad = nullptr;
         ibv_post_send(wq0, &w, &bad);
 
         // Wait up to 30 s for the send completion on NIC 0
         fd_set rfds; FD_ZERO(&rfds); FD_SET(cc->fd, &rfds);
         struct timeval tv = {30, 0};
         if (select(cc->fd + 1, &rfds, nullptr, nullptr, &tv) > 0) {
             struct ibv_cq* ev_cq; void* ev_ctx;
             if (ibv_get_cq_event(cc, &ev_cq, &ev_ctx) == 0)
                 ibv_ack_cq_events(ev_cq, 1);
             struct ibv_wc wc;
             ibv_poll_cq(wcq0, 1, &wc);
         }
     }
 
     if (wmr0) ibv_dereg_mr(wmr0);
     if (wmr1) ibv_dereg_mr(wmr1);
     free(wb0); free(wb1);
     ibv_destroy_qp(wq0); ibv_destroy_qp(wq1);
     ibv_destroy_cq(wcq0); ibv_destroy_cq(wcq1);
     ibv_destroy_comp_channel(cc);
 }
 
 // ── Test harness ──────────────────────────────────────────────────────────────
 
 struct MetricRow {
     int  trial;
     bool pass;
 };
 
 static std::pair<bool, std::vector<MetricRow>> runTest(
         struct ibv_device* dev0,
         struct ibv_device* dev1,
         int num_trials,
         int timeout_ms) {
 
     struct ibv_context* ctx0 = ibv_open_device(dev0);
     CHECK_PTR(ctx0, "ibv_open_device (NIC 0)");
     struct ibv_context* ctx1 = ibv_open_device(dev1);
     CHECK_PTR(ctx1, "ibv_open_device (NIC 1)");
 
     struct ibv_pd* pd0 = ibv_alloc_pd(ctx0);
     CHECK_PTR(pd0, "ibv_alloc_pd (NIC 0)");
     struct ibv_pd* pd1 = ibv_alloc_pd(ctx1);
     CHECK_PTR(pd1, "ibv_alloc_pd (NIC 1)");
 
     // Query port attributes and GIDs once, reuse across trials
     NicInfo n0 = queryNic(ctx0, pd0);
     NicInfo n1 = queryNic(ctx1, pd1);
 
     std::vector<MetricRow> rows;
     bool overall_pass = true;
 
     {
         // AsyncEventMonitor watches NIC 0's async_fd.
         // Inner scope: monitor is destroyed before ibv_dealloc_pd / ibv_close_device.
         AsyncEventMonitor monitor(ctx0);
         CHECK(monitor.setNonBlocking(),
               "Failed to set async_fd to non-blocking");
 
         // Verify O_NONBLOCK was actually set
         int fl = fcntl(monitor.getAsyncFd(), F_GETFL, 0);
         CHECK(fl != -1 && (fl & O_NONBLOCK), "async_fd O_NONBLOCK not set");
 
         // Warm-up: the first RDMA operation across a physical link triggers
         // hardware path initialization (ARP / neighbor resolution) that can
         // take several seconds.  Warm up once before timed trials.
         warmUpPath(n0, n1);
 
         for (int trial = 1; trial <= num_trials; trial++) {
             // Create fresh per-trial resources (tiny CQ + two QPs across two NICs).
             // Each trial needs a new CQ so we can observe a fresh overflow.
             TrialResources trial_res(n0, n1);
 
             // mlx5 uses cq->cqe as a ring-buffer mask: actual capacity = cqe+1.
             // Post cqe+2 = actual_capacity+1 writes to guarantee overflow.
             trial_res.postOverflowWrites(trial_res.getActualCqe() + 2);
 
             // Wait for the CQ overflow async event on NIC 0's async_fd
             struct ibv_async_event ev = {};
             bool got_event = monitor.waitEvent(&ev, timeout_ms);
 
             bool trial_pass = false;
 
             if (got_event) {
                 trial_pass = (ev.event_type == IBV_EVENT_CQ_ERR);
 
                 // Ack the CQ_ERR event — mandatory before ibv_destroy_cq()
                 monitor.ackEvent(&ev);
 
                 // Drain any extra events generated by the overflow.
                 // mlx5 may also deliver IBV_EVENT_QP_FATAL, IBV_EVENT_QP_REQ_ERR,
                 // or IBV_EVENT_QP_ACCESS_ERR alongside CQ_ERR depending on which
                 // QP side the error is detected on.  Ack all of them so that
                 // ibv_destroy_qp() does not block.
                 struct ibv_async_event extra = {};
                 while (monitor.waitEvent(&extra, 200))
                     monitor.ackEvent(&extra);
             } else {
                 // No event within timeout on mlx5 — should not happen.
                 // Treat as FAIL (CQ_ERR not received).
                 trial_pass = false;
             }
             // trial_res is destroyed here.  Safe because all events are acked.
 
             overall_pass = overall_pass && trial_pass;
             rows.push_back({trial, trial_pass});
         }
         // monitor destroyed here
     }
 
     ibv_dealloc_pd(pd1); ibv_dealloc_pd(pd0);
     ibv_close_device(ctx1); ibv_close_device(ctx0);
 
     return {overall_pass, rows};
 }
 
 // ── JSON output ───────────────────────────────────────────────────────────────
 
 static void printJson(bool overall_pass, const std::vector<MetricRow>& rows) {
     std::cout << "{\n";
     std::cout << "  \"Correctness\": \""
               << (overall_pass ? "PASS" : "FAIL") << "\",\n";
     // Each metrics entry uses "data_size" to index the trial number; declare
     // the unit explicitly to satisfy the dataset spec.
     std::cout << "  \"data_size_unit\": \"trial\",\n";
     std::cout << "  \"metrics\": [\n";
 
     for (size_t i = 0; i < rows.size(); i++) {
         const auto& r = rows[i];
         std::cout << "    {"
                   << "\"data_size\": " << r.trial
                   << ", \"pass\": "    << (r.pass ? "true" : "false")
                   << "}";
         if (i + 1 < rows.size()) std::cout << ",";
         std::cout << "\n";
     }
 
     std::cout << "  ]\n";
     std::cout << "}\n";
 }
 
 // ── Driver detection ──────────────────────────────────────────────────────────
 //
 // IBV_EVENT_CQ_ERR on CQ overflow is mlx5-specific behaviour.
 // rxe/siw silently drop overflow completions; Broadcom is unconfirmed.
 // Fail fast with a clear message rather than producing ambiguous results.
 //
 
 static bool isMLX5Device(struct ibv_device* device) {
     const char* name = ibv_get_device_name(device);
     return name && strncmp(name, "mlx5", 4) == 0;
 }
 
 // ── main ──────────────────────────────────────────────────────────────────────
 
 int main(int argc, char* argv[]) {
     int nic0_idx   = 0;    // NIC 0: sender — owns the tiny CQ and send QP
     int nic1_idx   = 1;    // NIC 1: receiver — owns the recv QP and remote buffer
     int num_trials = 5;
     int timeout_ms = 5000;
 
     if (argc > 1) nic0_idx   = std::atoi(argv[1]);
     if (argc > 2) nic1_idx   = std::atoi(argv[2]);
     if (argc > 3) num_trials = std::atoi(argv[3]);
     if (argc > 4) timeout_ms = std::atoi(argv[4]);
 
     int num_devices = 0;
     struct ibv_device** device_list = ibv_get_device_list(&num_devices);
     if (!device_list || num_devices == 0) {
         std::cerr << "No RDMA devices found\n";
         return EXIT_FAILURE;
     }
 
     if (nic0_idx >= num_devices || nic1_idx >= num_devices) {
         std::cerr << "Invalid NIC index (found " << num_devices << " devices)\n";
         ibv_free_device_list(device_list);
         return EXIT_FAILURE;
     }
 
     if (nic0_idx == nic1_idx) {
         std::cerr << "Error: NIC 0 and NIC 1 must be different devices "
                   << "(both are index " << nic0_idx << ").\n"
                   << "Specify two different NIC indices as argv[1] and argv[2].\n";
         ibv_free_device_list(device_list);
         return EXIT_FAILURE;
     }
 
     for (int idx : {nic0_idx, nic1_idx}) {
         if (!isMLX5Device(device_list[idx])) {
             std::cerr << "Error: device '"
                       << ibv_get_device_name(device_list[idx])
                       << "' (index " << idx << ") is not an mlx5 device.\n"
                       << "IBV_EVENT_CQ_ERR on CQ overflow requires the mlx5 driver.\n"
                       << "Supported: mlx5_0, mlx5_1, ... (ConnectX-4 and later)\n";
             ibv_free_device_list(device_list);
             return EXIT_FAILURE;
         }
     }
 
     std::cout << "NIC 0 (sender):   " << ibv_get_device_name(device_list[nic0_idx]) << "\n";
     std::cout << "NIC 1 (receiver): " << ibv_get_device_name(device_list[nic1_idx]) << "\n";
     std::cout << "Trials: " << num_trials << ", timeout: " << timeout_ms << " ms\n\n";
     // Flush before RDMA operations: libibverbs/mlx5 may fork() a child
     // process internally; any unflushed cout data would be duplicated when
     // both parent and child flush their buffers at exit.
     std::cout.flush();
 
     try {
         auto [overall_pass, rows] = runTest(
             device_list[nic0_idx], device_list[nic1_idx], num_trials, timeout_ms);
         ibv_free_device_list(device_list);
         printJson(overall_pass, rows);
         return overall_pass ? EXIT_SUCCESS : EXIT_FAILURE;
     } catch (const std::exception& e) {
         std::cerr << "Exception: " << e.what() << "\n";
         ibv_free_device_list(device_list);
         return EXIT_FAILURE;
     }
 }
 