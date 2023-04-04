#pragma once
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <iostream>
// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kINFO)
		: reportableSeverity(severity)
	{
	}

	void log(Severity severity, const char* msg) noexcept override
	{
		// suppress messages with severity enum value greater than the reportable
		if (severity > reportableSeverity)
			return;

		switch (severity)
		{
		case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
		case Severity::kERROR: std::cerr << "ERROR: "; break;
		case Severity::kWARNING: std::cerr << "WARNING: "; break;
		case Severity::kINFO: std::cerr << "INFO: "; break;
		default: std::cerr << "UNKNOWN: "; break;
		}
		std::cerr << msg << std::endl;
	}

	Severity getReportableSeverity() const{
		return reportableSeverity;
	}

private:
	Severity reportableSeverity;
};