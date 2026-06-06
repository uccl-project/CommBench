/*
 * RDMA NIC Information Query Tool
 *
 * This program queries local RDMA NIC information including device names,
 * vendors, GIDs, ports, and QP configurations using only <infiniband/verbs.h>.
 *
 * The RdmaNicInfo class provides methods to discover and query all RDMA
 * devices on the system, printing both human-readable summaries and
 * deterministic METRICS_JSON output for automated comparison.
 *
 * YOUR TASK: Implement the 9 methods marked with TODO below.
 * All output methods, data structures, includes, and main() are provided.
 * Do NOT modify the output methods or data structures.
 */

#include <infiniband/verbs.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

// ============================================================================
// Data Structures
// ============================================================================

struct PortInfo {
    uint8_t port_num;
    enum ibv_port_state state;
    enum ibv_mtu active_mtu;
    enum ibv_mtu max_mtu;
    uint8_t link_layer;
    uint8_t active_width;
    uint32_t active_speed;
    int gid_tbl_len;
    std::vector<union ibv_gid> gids;
};

struct DeviceInfo {
    std::string device_name;
    uint64_t node_guid;
    uint32_t vendor_id;
    uint32_t vendor_part_id;
    std::string fw_ver;
    uint8_t phys_port_cnt;
    int max_qp;
    int max_qp_wr;
    int max_cq;
    int max_cqe;
    int max_mr;
    int max_pd;
    int max_sge;
    enum ibv_atomic_cap atomic_cap;
    std::vector<PortInfo> ports;
};

// ============================================================================
// RdmaNicInfo Class
// ============================================================================

class RdmaNicInfo {
public:
    RdmaNicInfo() = default;
    ~RdmaNicInfo() = default;

    // ========================================================================
    // Discovery and Query Methods — YOU MUST IMPLEMENT THESE
    // ========================================================================

    /**
     * TODO 1: Discover all RDMA devices on the system.
     *
     * Use ibv_get_device_list(&num_devices) to get the device list.
     * Store the returned pointer in device_list_ and count in num_devices_.
     * Resize device_infos_ to num_devices and populate each device_infos_[i].device_name
     * using ibv_get_device_name(device_list[i]).
     * If no devices found, print error to stderr and return.
     */
    void discoverDevices() {
        // TODO: Implement device discovery
    }

    /**
     * TODO 2: Query device attributes for device at index idx.
     *
     * Steps:
     * 1. Bounds-check idx against num_devices_
     * 2. Open the device with ibv_open_device(device_list_[idx])
     * 3. Declare struct ibv_device_attr and zero it with memset
     * 4. Call ibv_query_device(ctx, &attr) to get device attributes
     * 5. Populate device_infos_[idx] fields:
     *    - node_guid, vendor_id, vendor_part_id, fw_ver, phys_port_cnt
     *    - max_qp, max_qp_wr, max_cq, max_cqe, max_mr, max_pd, max_sge
     *    - atomic_cap
     * 6. Close the device with ibv_close_device(ctx)
     */
    void queryDeviceAttributes(int idx) {
        // TODO: Implement device attribute query
    }

    /**
     * TODO 3: Query port info for device at index idx, port number port_num.
     *
     * IMPORTANT: Port numbers are 1-indexed in InfiniBand (first port is 1, not 0).
     *
     * Steps:
     * 1. Bounds-check idx
     * 2. Open the device with ibv_open_device(device_list_[idx])
     * 3. Declare struct ibv_port_attr and zero it
     * 4. Call ibv_query_port(ctx, port_num, &port_attr)
     * 5. Create a PortInfo struct and populate:
     *    - port_num, state, active_mtu, max_mtu, link_layer
     *    - active_width, active_speed, gid_tbl_len
     * 6. Loop from g=0 to port_attr.gid_tbl_len-1:
     *    - Call ibv_query_gid(ctx, port_num, g, &gid)
     *    - Skip zero GIDs (all 16 bytes are 0)
     *    - Push non-zero GIDs into pinfo.gids
     * 7. Push the PortInfo into device_infos_[idx].ports
     * 8. Close the device
     */
    void queryPortInfo(int idx, uint8_t port_num) {
        // TODO: Implement port info query
    }

    /**
     * TODO 4: Query all ports for device at index idx.
     *
     * Iterate from port 1 to device_infos_[idx].phys_port_cnt (inclusive).
     * Call queryPortInfo(idx, port_num) for each port.
     * Note: phys_port_cnt must already be populated by queryDeviceAttributes().
     */
    void queryAllPorts(int idx) {
        // TODO: Implement all-ports query
    }

    /**
     * TODO 5: Orchestrate full discovery and query.
     *
     * Steps:
     * 1. Call discoverDevices()
     * 2. For each device i from 0 to num_devices_-1:
     *    a. Call queryDeviceAttributes(i)
     *    b. Call queryAllPorts(i)
     */
    void queryAll() {
        // TODO: Implement full query orchestration
    }

    /**
     * TODO 6: Format a GID as colon-separated hex string.
     *
     * Format: "xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx" (8 groups of 4 hex chars)
     * Each group is two consecutive bytes from gid.raw[].
     * Use snprintf with %02x format for each byte.
     * Example: "fe80:0000:0000:0000:0a0b:0c0d:0e0f:1011"
     */
    static std::string formatGid(const union ibv_gid& gid) {
        // TODO: Implement GID formatting
        return "";
    }

    /**
     * TODO 7: Convert IBV_MTU enum to string.
     *
     * Mapping:
     *   IBV_MTU_256  -> "256"
     *   IBV_MTU_512  -> "512"
     *   IBV_MTU_1024 -> "1024"
     *   IBV_MTU_2048 -> "2048"
     *   IBV_MTU_4096 -> "4096"
     *   default      -> "unknown"
     */
    static std::string mtuToString(enum ibv_mtu mtu) {
        // TODO: Implement MTU to string conversion
        return "";
    }

    /**
     * TODO 8: Convert link layer type to string.
     *
     * Mapping:
     *   IBV_LINK_LAYER_UNSPECIFIED -> "Unspecified"
     *   IBV_LINK_LAYER_INFINIBAND  -> "InfiniBand"
     *   IBV_LINK_LAYER_ETHERNET    -> "Ethernet"
     *   default                    -> "Unknown"
     */
    static std::string linkLayerToString(uint8_t link_layer) {
        // TODO: Implement link layer to string conversion
        return "";
    }

    /**
     * TODO 9: Convert atomic capability enum to string.
     *
     * Mapping:
     *   IBV_ATOMIC_NONE -> "NONE"
     *   IBV_ATOMIC_HCA  -> "HCA"
     *   IBV_ATOMIC_GLOB -> "GLOB"
     *   default         -> "UNKNOWN"
     */
    static std::string atomicCapToString(enum ibv_atomic_cap cap) {
        // TODO: Implement atomic cap to string conversion
        return "";
    }

    // ========================================================================
    // Output Methods (given — DO NOT MODIFY)
    // ========================================================================

    /**
     * Print human-readable device summary.
     */
    void printDeviceSummary() const {
        for (int i = 0; i < (int)device_infos_.size(); i++) {
            const DeviceInfo& dev = device_infos_[i];
            std::cout << "========================================" << std::endl;
            std::cout << "Device " << i << ": " << dev.device_name << std::endl;
            std::cout << "========================================" << std::endl;

            // Format node GUID
            uint64_t guid = dev.node_guid;
            char guid_str[32];
            snprintf(guid_str, sizeof(guid_str),
                     "%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x",
                     (unsigned)((guid >> 56) & 0xFF),
                     (unsigned)((guid >> 48) & 0xFF),
                     (unsigned)((guid >> 40) & 0xFF),
                     (unsigned)((guid >> 32) & 0xFF),
                     (unsigned)((guid >> 24) & 0xFF),
                     (unsigned)((guid >> 16) & 0xFF),
                     (unsigned)((guid >> 8) & 0xFF),
                     (unsigned)(guid & 0xFF));

            std::cout << "  Node GUID:       " << guid_str << std::endl;
            std::cout << "  Vendor ID:       0x" << std::hex << std::setfill('0')
                      << std::setw(4) << dev.vendor_id << std::dec << std::endl;
            std::cout << "  Vendor Part ID:  0x" << std::hex << std::setfill('0')
                      << std::setw(4) << dev.vendor_part_id << std::dec << std::endl;
            std::cout << "  Firmware Ver:    " << dev.fw_ver << std::endl;
            std::cout << "  Phys Ports:      " << (int)dev.phys_port_cnt << std::endl;
            std::cout << "  Max QP:          " << dev.max_qp << std::endl;
            std::cout << "  Max QP WR:       " << dev.max_qp_wr << std::endl;
            std::cout << "  Max CQ:          " << dev.max_cq << std::endl;
            std::cout << "  Max CQE:         " << dev.max_cqe << std::endl;
            std::cout << "  Max MR:          " << dev.max_mr << std::endl;
            std::cout << "  Max PD:          " << dev.max_pd << std::endl;
            std::cout << "  Max SGE:         " << dev.max_sge << std::endl;
            std::cout << "  Atomic Cap:      " << atomicCapToString(dev.atomic_cap) << std::endl;

            for (const auto& port : dev.ports) {
                std::cout << std::endl;
                std::cout << "  Port " << (int)port.port_num << ":" << std::endl;
                std::cout << "    State:         " << portStateToString(port.state) << std::endl;
                std::cout << "    Active MTU:    " << mtuToString(port.active_mtu) << std::endl;
                std::cout << "    Max MTU:       " << mtuToString(port.max_mtu) << std::endl;
                std::cout << "    Link Layer:    " << linkLayerToString(port.link_layer) << std::endl;
                std::cout << "    Active Width:  " << (int)port.active_width << std::endl;
                std::cout << "    Active Speed:  " << port.active_speed << std::endl;
                std::cout << "    GID Table Len: " << port.gid_tbl_len << std::endl;
                std::cout << "    Non-zero GIDs: " << port.gids.size() << std::endl;
                for (size_t g = 0; g < port.gids.size(); g++) {
                    std::cout << "      GID[" << g << "]: " << formatGid(port.gids[g]) << std::endl;
                }
            }
        }
    }

    /**
     * Print deterministic METRICS_JSON line per device.
     * Fields are alphabetically sorted for deterministic comparison.
     */
    void printMetricsJson() const {
        for (const auto& dev : device_infos_) {
            std::ostringstream oss;
            oss << "{";

            // Required by the dataset spec: every JSON object must declare
            // a "Correctness" field. This example only enumerates NIC
            // attributes, so success of the enumeration itself is the
            // correctness signal.
            oss << "\"Correctness\":\"PASS\", ";

            // Fields in alphabetical order
            oss << "\"atomic_cap\":\"" << atomicCapToString(dev.atomic_cap) << "\"";
            oss << ", \"device_name\":\"" << dev.device_name << "\"";
            oss << ", \"fw_ver\":\"" << dev.fw_ver << "\"";
            oss << ", \"max_cq\":" << dev.max_cq;
            oss << ", \"max_cqe\":" << dev.max_cqe;
            oss << ", \"max_mr\":" << dev.max_mr;
            oss << ", \"max_pd\":" << dev.max_pd;
            oss << ", \"max_qp\":" << dev.max_qp;
            oss << ", \"max_qp_wr\":" << dev.max_qp_wr;
            oss << ", \"max_sge\":" << dev.max_sge;
            oss << ", \"num_ports\":" << (int)dev.phys_port_cnt;

            // Port-specific fields
            for (const auto& port : dev.ports) {
                std::string prefix = "port_" + std::to_string(port.port_num);
                oss << ", \"" << prefix << "_active_mtu\":\"" << mtuToString(port.active_mtu) << "\"";
                oss << ", \"" << prefix << "_active_speed\":" << port.active_speed;
                oss << ", \"" << prefix << "_active_width\":" << (int)port.active_width;
                oss << ", \"" << prefix << "_gid_count\":" << port.gids.size();
                oss << ", \"" << prefix << "_link_layer\":\"" << linkLayerToString(port.link_layer) << "\"";
                oss << ", \"" << prefix << "_state\":\"" << portStateToString(port.state) << "\"";
            }

            // vendor_id and vendor_part_id as hex strings
            char vid[16], vpid[16];
            snprintf(vid, sizeof(vid), "0x%04x", dev.vendor_id);
            snprintf(vpid, sizeof(vpid), "0x%04x", dev.vendor_part_id);
            oss << ", \"vendor_id\":\"" << vid << "\"";
            oss << ", \"vendor_part_id\":\"" << vpid << "\"";

            oss << "}";
            std::cout << oss.str() << std::endl;
        }
    }

    /**
     * Print all information: summary + metrics JSON.
     */
    void printAll() const {
        printDeviceSummary();
        std::cout << std::endl;
        printMetricsJson();
    }

    /**
     * Get number of discovered devices.
     */
    int getDeviceCount() const {
        return (int)device_infos_.size();
    }

    /**
     * Get device info by index.
     */
    const DeviceInfo& getDevice(int idx) const {
        return device_infos_[idx];
    }

private:
    /**
     * Convert port state enum to string.
     */
    static std::string portStateToString(enum ibv_port_state state) {
        switch (state) {
            case IBV_PORT_NOP:          return "NOP";
            case IBV_PORT_DOWN:         return "PORT_DOWN";
            case IBV_PORT_INIT:         return "PORT_INIT";
            case IBV_PORT_ARMED:        return "PORT_ARMED";
            case IBV_PORT_ACTIVE:       return "PORT_ACTIVE";
            case IBV_PORT_ACTIVE_DEFER: return "PORT_ACTIVE_DEFER";
            default:                    return "UNKNOWN";
        }
    }

    struct ibv_device** device_list_ = nullptr;
    int num_devices_ = 0;
    std::vector<DeviceInfo> device_infos_;
};

// ============================================================================
// Main
// ============================================================================

int main() {
    RdmaNicInfo nic_info;

    try {
        nic_info.queryAll();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "FAIL" << std::endl;
        return EXIT_FAILURE;
    }

    if (nic_info.getDeviceCount() == 0) {
        std::cerr << "No RDMA devices found" << std::endl;
        std::cerr << "FAIL" << std::endl;
        return EXIT_FAILURE;
    }

    nic_info.printAll();

    std::cout << std::endl;
    std::cout << "PASSED" << std::endl;
    return EXIT_SUCCESS;
}
