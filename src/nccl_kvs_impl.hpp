#pragma once

#ifdef CCL_ENABLE_NCCL

#include "oneapi/ccl/types.hpp"
#include "oneapi/ccl/aliases.hpp"
#include "common/global/global.hpp"
#include "common/api_wrapper/nccl_api_wrapper.hpp"
#include "kvs_impl.hpp"

namespace ccl {

class nccl_kvs_impl : public base_kvs_impl {
public:
    // Costruttore per il rank 0 (main KVS) - genera l'ncclUniqueId
    nccl_kvs_impl();
    
    // Costruttore per gli altri rank - riceve l'address con l'ncclUniqueId
    nccl_kvs_impl(const kvs::address_type& addr);
    
    // Ottieni l'ncclUniqueId
    ncclUniqueId get_nccl_id() const;
    
    // get/set non servono per NCCL (il KVS è solo per distribuire l'ID)
    vector_class<char> get(const string_class& key) override {
        CCL_THROW("get() is not needed for NCCL backend");
    }
    
    void set(const string_class& key, const vector_class<char>& data) override {
        CCL_THROW("set() is not needed for NCCL backend");
    }

private:
    ncclUniqueId nccl_id;
    
    // Helper per convertire ncclUniqueId <-> address_type
    static kvs::address_type convert_id_to_addr(const ncclUniqueId& id);
    static ncclUniqueId convert_addr_to_id(const kvs::address_type& addr);
};

} // namespace ccl

#endif // CCL_ENABLE_NCCL
