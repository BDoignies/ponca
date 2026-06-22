set(PYPONCA_SRC
    pyponca.cpp
)

nanobind_add_module(pyponca ${PYPONCA_SRC})
