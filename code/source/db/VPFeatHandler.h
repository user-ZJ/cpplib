//
// SQLExecutor.h
//
// Definition of the SQLExecutor class.
//
// Copyright (c) 2008, Applied Informatics Software Engineering GmbH.
// and Contributors.
//
// SPDX-License-Identifier:	BSL-1.0
//

#ifndef SQLExecutor_INCLUDED
#define SQLExecutor_INCLUDED

#include "Poco/Data/MySQL/MySQL.h"
#include "Poco/Data/Session.h"
#include "Poco/Timestamp.h"

namespace BASE_NAMESPACE {

struct VPFeat {
  int id = 0;  // 自增键，传0会自增
  std::string speaker_id;
  std::string dmid;
  Poco::Data::CLOB feature;
  Poco::DateTime create_time;
  Poco::DateTime update_time;
  int soft_del = 0;
  VPFeat() = default;
  VPFeat(const std::string &spk_id, const std::string &dmid, const Poco::Data::CLOB &feat,
         Poco::DateTime ct = Poco::DateTime(), Poco::DateTime ut = Poco::DateTime()) :
    dmid(dmid),
    speaker_id(spk_id), feature(feat), create_time(ct), update_time(ut) {}

  VPFeat(const std::string &spk_id, const std::string &dmid, const std::vector<char> &feat,
         Poco::DateTime ct = Poco::DateTime(), Poco::DateTime ut = Poco::DateTime()) :
    dmid(dmid),
    speaker_id(spk_id), feature(feat), create_time(ct), update_time(ut) {}

  const std::string &operator()() const
  /// This method is required so we can extract data to a map!
  {
    // we choose the lastName as examplary key
    return speaker_id;
  }
};

}  // namespace BASE_NAMESPACE

namespace Poco { namespace Data {

template <>
class TypeHandler<BASE_NAMESPACE::VPFeat> {
 public:
  static void bind(std::size_t pos, const BASE_NAMESPACE::VPFeat &obj, AbstractBinder::Ptr pBinder,
                   AbstractBinder::Direction dir) {
    // poco_assert_dbg (!pBinder.isNull());
    pBinder->bind(pos++, obj.id, dir);
    pBinder->bind(pos++, obj.speaker_id, dir);
    pBinder->bind(pos++, obj.dmid, dir);
    pBinder->bind(pos++, obj.feature, dir);
    pBinder->bind(pos++, obj.create_time, dir);
    pBinder->bind(pos++, obj.update_time, dir);
    pBinder->bind(pos++, obj.soft_del, dir);
  }

  static void prepare(std::size_t pos, const BASE_NAMESPACE::VPFeat &obj, AbstractPreparator::Ptr pPrepare) {
    // poco_assert_dbg (!pPrepare.isNull());
    pPrepare->prepare(pos++, obj.id);
    pPrepare->prepare(pos++, obj.speaker_id);
    pPrepare->prepare(pos++, obj.dmid);
    pPrepare->prepare(pos++, obj.feature);
    pPrepare->prepare(pos++, obj.create_time);
    pPrepare->prepare(pos++, obj.update_time);
    pPrepare->prepare(pos++, obj.soft_del);
  }

  static std::size_t size() {
    return 6;
  }

  static void extract(std::size_t pos, BASE_NAMESPACE::VPFeat &obj, const BASE_NAMESPACE::VPFeat &defVal, AbstractExtractor::Ptr pExt) {
    // poco_assert_dbg (!pExt.isNull());
    if (!pExt->extract(pos++, obj.id)) obj.id = defVal.id;
    if (!pExt->extract(pos++, obj.speaker_id)) obj.speaker_id = defVal.speaker_id;
    if (!pExt->extract(pos++, obj.dmid)) obj.dmid = defVal.dmid;
    if (!pExt->extract(pos++, obj.feature)) obj.feature = defVal.feature;
    if (!pExt->extract(pos++, obj.create_time)) obj.create_time = defVal.create_time;
    if (!pExt->extract(pos++, obj.update_time)) obj.update_time = defVal.update_time;
    if (!pExt->extract(pos++, obj.soft_del)) obj.soft_del = defVal.soft_del;
  }

 private:
  TypeHandler();
  ~TypeHandler();
  TypeHandler(const TypeHandler &);
  TypeHandler &operator=(const TypeHandler &);
};

}}  // namespace Poco::Data

#endif  // SQLExecutor_INCLUDED
