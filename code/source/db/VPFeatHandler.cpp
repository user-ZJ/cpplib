//
// SQLExecutor.cpp
//
// Copyright (c) 2008, Applied Informatics Software Engineering GmbH.
// and Contributors.
//
// SPDX-License-Identifier:	BSL-1.0
//

#include "VPFeatHandler.h"
#include "Poco/Any.h"
#include "Poco/Data/Date.h"
#include "Poco/Data/LOB.h"
#include "Poco/Data/MySQL/Connector.h"
#include "Poco/Data/MySQL/MySQLException.h"
#include "Poco/Data/RecordSet.h"
#include "Poco/Data/StatementImpl.h"
#include "Poco/Data/Time.h"
#include "Poco/Data/Transaction.h"
#include "Poco/DateTime.h"
#include "Poco/Exception.h"
#include "Poco/Format.h"
#include "Poco/String.h"
#include "Poco/Timestamp.h"
#include "Poco/Tuple.h"
#include "utils/logging.h"

#ifdef _WIN32
#include <Winsock2.h>
#endif

#include <iostream>
#include <limits>
#include <mysql/mysql.h>

using namespace Poco::Data;
using namespace Poco::Data::Keywords;
using Poco::Any;
using Poco::AnyCast;
using Poco::BadCastException;
using Poco::DateTime;
using Poco::format;
using Poco::InvalidAccessException;
using Poco::NotFoundException;
using Poco::RangeException;
using Poco::Timestamp;
using Poco::Tuple;
using Poco::Data::MySQL::ConnectionException;
using Poco::Data::MySQL::StatementException;

namespace BASE_NAMESPACE {

}  // namespace BASE_NAMESPACE
