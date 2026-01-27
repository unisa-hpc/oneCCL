#ifdef CCL_ENABLE_NCCL

#include "nccl_kvs_impl.hpp"
#include "common/log/log.hpp"
#include <cstring>

namespace ccl {

// Converte ncclUniqueId (128 byte) -> address_type (256 byte)
kvs::address_type nccl_kvs_impl::convert_id_to_addr(const ncclUniqueId& id) {
    kvs::address_type addr{ 0 };
    
    // ncclUniqueId è un array di byte, copialo nell'address
    static_assert(sizeof(ncclUniqueId) <= kvs::address_max_size, 
                  "ncclUniqueId too large for kvs::address_type");
    
    memcpy(addr.data(), &id, sizeof(ncclUniqueId));
    
    return addr;
}

// Converte address_type -> ncclUniqueId
ncclUniqueId nccl_kvs_impl::convert_addr_to_id(const kvs::address_type& addr) {
    ncclUniqueId id;
    memcpy(&id, addr.data(), sizeof(ncclUniqueId));
    return id;
}

// Costruttore per rank 0 (main KVS)
nccl_kvs_impl::nccl_kvs_impl() : base_kvs_impl() {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::nccl, 
                     "unexpected backend");
    
    // Genera un nuovo ncclUniqueId
    auto status = ncclGetUniqueId(&nccl_id);
    CCL_THROW_IF_NOT(status == ncclSuccess, 
                     "ncclGetUniqueId failed: ", ncclGetErrorString(status));
    
    LOG_DEBUG("NCCL KVS: Generated unique ID");
    
    // Salva l'ID nell'address
    addr = convert_id_to_addr(nccl_id);
}

// Costruttore per altri rank
nccl_kvs_impl::nccl_kvs_impl(const kvs::address_type& input_addr) 
    : base_kvs_impl(input_addr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::nccl, 
                     "unexpected backend");
    
    // Estrai l'ncclUniqueId dall'address
    nccl_id = convert_addr_to_id(input_addr);
    
    LOG_DEBUG("NCCL KVS: Received unique ID from address");
}

ncclUniqueId nccl_kvs_impl::get_nccl_id() const {
    return nccl_id;
}

} // namespace ccl

#endif // CCL_ENABLE_NCCL
