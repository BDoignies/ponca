/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#ifdef _PONCA_SHARED_LIBRARY
#    define _INSTANTIATE_PONCA_EXTERN
#    ifdef _MSC_VER
#        define _PONCA_API __declspec(dllexport)
#    else
#        define _PONCA_API 
#    endif
#else
#    define _INSTANTIATE_PONCA_EXTERN extern
#    ifdef _MSC_VER
#        define _PONCA_API __declspec(dllimport)
#    else
#        define _PONCA_API 
#    endif
#endif

#include "../../../Ponca"

namespace Ponca
{
// Note: Variadic macros here because ',' in template definition are parsed as different arguments
// we could use a trick by wrapping the class in parenthesis, but for now this works and is simpler.
#define _PONCA_INSTANTIATE(...) _INSTANTIATE_PONCA_EXTERN template class _PONCA_API __VA_ARGS__;
#define _PONCA_BASKET_X(name, desc, ...) _PONCA_INSTANTIATE(__VA_ARGS__);
#define _PONCA_BASKET_DIFF_X(name, desc, ...) _PONCA_INSTANTIATE(__VA_ARGS__);
#include "instantiate_list/entrypoint.h"
#undef _PONCA_BASKET_X
#undef _PONCA_BASKET_DIFF_X
#undef _PONCA_INSTANTIATE
}; // namespace Ponca
