#pragma once

#if defined(_WIN32) || defined(_WIN64)
#if defined(PTEX_BUILD_DLL)
#define PTEX_API __declspec(dllexport)
#elif defined(PTEX_USE_DLL)
#define PTEX_API __declspec(dllimport)
#else
#define PTEX_API
#endif
#else
#define PTEX_API
#endif