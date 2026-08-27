//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef RIOT_UTILS_PY_INIT_HPP_
#define RIOT_UTILS_PY_INIT_HPP_
// This file was made in part with generative AI.

#include <Python.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// ------------ Python / NumPy bootstrap ------------

struct PyException : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string fetch_python_error() {
  PyObject *ptype = nullptr, *pvalue = nullptr, *ptrace = nullptr;
  PyErr_Fetch(&ptype, &pvalue, &ptrace);
  PyErr_NormalizeException(&ptype, &pvalue, &ptrace);
  PyObject *str_obj = pvalue ? PyObject_Str(pvalue) : nullptr;
  const char *msg = (str_obj && PyUnicode_Check(str_obj)) ? PyUnicode_AsUTF8(str_obj)
                                                          : "Unknown Python error";
  std::string out = msg ? msg : "Unknown Python error";
  Py_XDECREF(str_obj);
  Py_XDECREF(ptype);
  Py_XDECREF(pvalue);
  Py_XDECREF(ptrace);
  return out;
}

struct GILGuard {
  PyGILState_STATE st;
  GILGuard() : st(PyGILState_Ensure()) {}
  ~GILGuard() { PyGILState_Release(st); }
};

class PyPtr {
  PyObject *p_ = nullptr;

 public:
  PyPtr() = default;
  explicit PyPtr(PyObject *p) : p_(p) {}
  ~PyPtr() { Py_XDECREF(p_); }
  PyPtr(const PyPtr &) = delete;
  PyPtr &operator=(const PyPtr &) = delete;
  PyPtr(PyPtr &&o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  PyPtr &operator=(PyPtr &&o) noexcept {
    if (this != &o) {
      Py_XDECREF(p_);
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  PyObject *get() const { return p_; }
  PyObject *release() {
    PyObject *t = p_;
    p_ = nullptr;
    return t;
  }
  explicit operator bool() const { return p_ != nullptr; }
};

class PythonEnv {
 public:
  static void initialize(const std::vector<std::string> &extra_sys_path = {}) {
    static std::once_flag flag;
    std::call_once(flag, [&]() {
      if (!Py_IsInitialized()) {
        Py_Initialize();
      }
      // Ensure threading support and acquire GIL for this thread.
#if PY_VERSION_HEX < 0x03090000
      PyEval_InitThreads();
#endif // PY_VERSION_HEX < 0x03090000
      {
        GILGuard gil;
        // Optionally push extra sys.path entries (e.g., where your modules live)
        if (!extra_sys_path.empty()) {
          PyPtr sys(PyImport_ImportModule("sys"));
          if (!sys) throw PyException("Failed to import sys");
          PyObject *path = PyObject_GetAttrString(sys.get(), "path"); // borrowed
          for (const auto &p : extra_sys_path) {
            PyPtr pystr(PyUnicode_FromString(p.c_str()));
            if (!pystr) throw PyException("Failed to build sys.path string");
            if (PyList_Append(path, pystr.get()) != 0) {
              throw PyException("Failed to append to sys.path");
            }
          }
        }
      }
    });
  }

  static PyObject *numpy_module() {
    // Import and cache numpy at runtime (no NumPy C-API macros needed)
    static std::once_flag once;
    static PyObject *np = nullptr;
    std::call_once(once, []() {
      GILGuard gil;
      np = PyImport_ImportModule("numpy");
      if (!np) {
        throw PyException("Failed to import numpy: " + fetch_python_error());
      }
      // Leak on purpose until program end; interpreter owns it. (or wrap in PyPtr and
      // never release)
    });
    return np; // borrowed
  }
};

namespace parthenon {
class ParameterInput;
}
namespace Python {
void Init(parthenon::ParameterInput *pin);
} // namespace Python

#endif // RIOT_UTILS_PY_INIT_HPP_
