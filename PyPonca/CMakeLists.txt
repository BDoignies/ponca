set(PYPONCA_SRC
    binding/pyponca.cpp
)

nanobind_add_module(_pyponca ${PYPONCA_SRC})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_pyponca PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(_pyponca PUBLIC PYPONCA_WITH_OMP)
endif()

# We link here manually with stdc++. Nanobind seems to think this 
# is C-code when linking with our interface module...???
target_link_libraries(_pyponca PUBLIC Common Fitting stdc++)

install(TARGETS _pyponca DESTINATION pyponca)
install(FILES pyponca/__init__.py DESTINATION pyponca)