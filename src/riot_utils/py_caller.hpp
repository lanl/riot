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
#ifndef RIOT_UTILS_PY_CALLER_HPP_
#define RIOT_UTILS_PY_CALLER_HPP_
// This file was made in part with generative AI.

#include <Python.h>
#include <cstring>
#include <iomanip>
#include <ios>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "riot_utils/py_init.hpp"

#if !defined(PCALL_ENABLE_NUMPY_CAPI)
#define PCALL_ENABLE_NUMPY_CAPI 1 // set to 0 to force copy-path only
#endif

#if PCALL_ENABLE_NUMPY_CAPI
#ifndef PY_ARRAY_UNIQUE_SYMBOL
#define PY_ARRAY_UNIQUE_SYMBOL PCALL_NUMPY_API
#endif

#ifndef NO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY 1
#endif

// Use the modern NumPy C-API
#ifndef NPY_NO_DEPRECATED_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#endif
#include <numpy/arrayobject.h>
#endif

#if PY_VERSION_HEX >= 0x03080000
#define PCALL_HAVE_VECTORCALL 1
#else
#define PCALL_HAVE_VECTORCALL 0
#endif

// Vectorcall-on-method (PyObject_VectorcallMethod) landed in 3.9
#if PY_VERSION_HEX >= 0x03090000
#define PCALL_HAVE_VECTORCALL_METHOD 1
#else
#define PCALL_HAVE_VECTORCALL_METHOD 0
#endif

namespace pcall {

// ------------ small RAII / utilities ------------

// Generic product
inline size_t _prod(const std::vector<size_t> &v) {
  return std::accumulate(v.begin(), v.end(), size_t{1}, std::multiplies<size_t>());
}

#if PCALL_ENABLE_NUMPY_CAPI
inline std::vector<npy_intp> _to_npy_shape(const std::vector<size_t> &shp) {
  std::vector<npy_intp> out(shp.size());
  for (size_t i = 0; i < shp.size(); ++i)
    out[i] = static_cast<npy_intp>(shp[i]);
  return out;
}
inline std::vector<npy_intp> _c_strides(const std::vector<npy_intp> &shape,
                                        size_t itemsize) {
  std::vector<npy_intp> strides(shape.size());
  npy_intp s = static_cast<npy_intp>(itemsize);
  for (ptrdiff_t i = static_cast<ptrdiff_t>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = s;
    s *= shape[static_cast<size_t>(i)];
  }
  return strides;
}
#endif

inline bool _looks_like_path(const std::string &s) {
  auto ends_with = [&](const char *suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
  };
  return s.find('/') != std::string::npos || s.find('\\') != std::string::npos ||
         ends_with(".py") || ends_with(".pyw") || ends_with(".pyc") ||
         ends_with(".pyo") || ends_with(".so") || ends_with(".pyd") || ends_with(".dll");
}

// simple FNV-1a 64-bit for stable, cross-platform hashing
inline uint64_t _fnv1a64(const char *p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<unsigned char>(p[i]);
    h *= 1099511628211ull;
  }
  return h;
}
inline std::string _hex64(uint64_t x, int digits = 10) {
  std::ostringstream os;
  os << std::hex << std::nouppercase << std::setfill('0') << std::setw(digits)
     << (x & ((1ull << (4 * digits)) - 1));
  return os.str();
}

/* Import by module name OR by file path.
 * - If `import_as` is non-empty, that name is used in sys.modules.
 * - If empty, we derive a unique name: "_pcall_<stem>_<hash10>"
 * Returns new reference to module object.
 */
inline PyObject *
_import_module_by_name_or_file(const std::string &mod_or_path,
                               const std::string &import_as = std::string()) {
  if (!_looks_like_path(mod_or_path)) {
    // Import by dotted name
    return PyImport_ImportModule(mod_or_path.c_str()); // new ref or nullptr
  }

  // ---------- Import from a file path ----------
  // Python imports we'll need
  PyPtr sys(PyImport_ImportModule("sys"));
  PyPtr importlib_util(PyImport_ImportModule("importlib.util"));
  PyPtr osmod(PyImport_ImportModule("os"));
  if (!sys || !importlib_util || !osmod) {
    throw PyException("Failed to import 'sys'/'importlib.util'/'os': " +
                      fetch_python_error());
  }

  PyObject *os_path = PyObject_GetAttrString(osmod.get(), "path"); // borrowed
  if (!os_path) throw PyException("os.path missing");

  // abspath(path)
  PyPtr py_path(PyUnicode_FromString(mod_or_path.c_str()));
  PyPtr abspath(PyObject_GetAttrString(os_path, "abspath"));
  PyPtr abs_res(PyObject_CallFunctionObjArgs(abspath.get(), py_path.get(), nullptr));
  if (!abs_res) throw PyException("os.path.abspath failed: " + fetch_python_error());
  const char *abs_c = PyUnicode_AsUTF8(abs_res.get());
  if (!abs_c) throw PyException("abs path decode failed: " + fetch_python_error());
  std::string abspath_str(abs_c);

  // If it's a directory and has __init__.py, treat that as the module file
  PyPtr isdir(PyObject_GetAttrString(os_path, "isdir"));
  PyPtr exists(PyObject_GetAttrString(os_path, "exists"));
  PyPtr join(PyObject_GetAttrString(os_path, "join"));
  PyPtr dir_q(PyObject_CallFunctionObjArgs(isdir.get(), abs_res.get(), nullptr));
  bool is_dir = (dir_q && PyObject_IsTrue(dir_q.get()));
  if (is_dir) {
    PyPtr init_name(PyUnicode_FromString("__init__.py"));
    PyPtr init_path(PyObject_CallFunctionObjArgs(join.get(), abs_res.get(),
                                                 init_name.get(), nullptr));
    if (!init_path) throw PyException("join failed: " + fetch_python_error());
    PyPtr init_exists(
        PyObject_CallFunctionObjArgs(exists.get(), init_path.get(), nullptr));
    if (init_exists && PyObject_IsTrue(init_exists.get())) {
      abs_res = std::move(init_path); // use __init__.py
      const char *tmp = PyUnicode_AsUTF8(abs_res.get());
      if (!tmp) throw PyException("init path decode failed: " + fetch_python_error());
      abspath_str.assign(tmp);
    } else {
      throw PyException("Directory given but no __init__.py found");
    }
  }

  // Derive <stem> from basename (without extension)
  PyPtr basename(PyObject_GetAttrString(os_path, "basename"));
  PyPtr splitext(PyObject_GetAttrString(os_path, "splitext"));
  PyPtr base(PyObject_CallFunctionObjArgs(basename.get(), abs_res.get(), nullptr));
  if (!base) throw PyException("basename failed: " + fetch_python_error());
  PyPtr pair(PyObject_CallFunctionObjArgs(splitext.get(), base.get(), nullptr));
  if (!pair) throw PyException("splitext failed: " + fetch_python_error());
  PyObject *stem_obj = PyTuple_GetItem(pair.get(), 0); // borrowed
  if (!stem_obj) throw PyException("splitext tuple error");
  const char *stem_c = PyUnicode_AsUTF8(stem_obj);
  if (!stem_c) throw PyException("stem decode failed: " + fetch_python_error());
  std::string stem(stem_c);

  // Choose module name
  std::string modname = import_as.empty()
                            ? (std::string("_pcall_") + stem + "_" +
                               _hex64(_fnv1a64(abspath_str.c_str(), abspath_str.size())))
                            : import_as;

  // Reuse from sys.modules if present
  PyObject *modules = PyObject_GetAttrString(sys.get(), "modules"); // borrowed
  if (!modules) throw PyException("sys.modules missing");
  PyPtr py_modname(PyUnicode_FromString(modname.c_str()));
  PyObject *existing = PyDict_GetItem(modules, py_modname.get()); // borrowed
  if (existing) {
    Py_INCREF(existing);
    return existing;
  } // return existing module

  // spec = importlib.util.spec_from_file_location(modname, abspath)
  PyPtr spec_from_file(
      PyObject_GetAttrString(importlib_util.get(), "spec_from_file_location"));
  PyPtr module_from_spec(
      PyObject_GetAttrString(importlib_util.get(), "module_from_spec"));
  if (!spec_from_file || !module_from_spec)
    throw PyException("importlib.util API missing");
  PyPtr spec(PyObject_CallFunctionObjArgs(spec_from_file.get(), py_modname.get(),
                                          abs_res.get(), nullptr));
  if (!spec) throw PyException("spec_from_file_location failed: " + fetch_python_error());

  // mod = importlib.util.module_from_spec(spec)
  PyPtr module(PyObject_CallFunctionObjArgs(module_from_spec.get(), spec.get(), nullptr));
  if (!module) throw PyException("module_from_spec failed: " + fetch_python_error());

  // spec.loader.exec_module(mod)
  PyPtr loader(PyObject_GetAttrString(spec.get(), "loader"));
  if (!loader) throw PyException("spec.loader missing");
  PyPtr exec_name(PyUnicode_FromString("exec_module"));
  PyPtr res(
      PyObject_CallMethodObjArgs(loader.get(), exec_name.get(), module.get(), nullptr));
  if (!res) throw PyException("loader.exec_module failed: " + fetch_python_error());

  // Register in sys.modules and return
  if (PyDict_SetItem(modules, py_modname.get(), module.get()) != 0)
    throw PyException("Failed to register module in sys.modules: " +
                      fetch_python_error());

  return module.release(); // new ref
}

// ------------ type traits & dtype mapping ------------
template <class T>
struct is_supported_vector_elem
    : std::bool_constant<
          std::is_same_v<T, double> || std::is_same_v<T, float> ||
          std::is_same_v<T, int64_t> || std::is_same_v<T, bool> ||
          // any 32-bit signed integral (covers int32_t and int when 32-bit)
          (std::is_integral_v<T> && !std::is_same_v<T, bool> && sizeof(T) == 4)> {};

template <class T>
struct numpy_dtype_string {
  static constexpr const char *value =
      std::is_same_v<T, double>    ? "float64"
      : std::is_same_v<T, float>   ? "float32"
      : std::is_same_v<T, int64_t> ? "int64"
      : std::is_same_v<T, bool>    ? "bool_"
      : (std::is_integral_v<T> && !std::is_same_v<T, bool> && sizeof(T) == 4)
          ? "int32"
          : (const char *)nullptr;
};
static_assert(numpy_dtype_string<double>::value, "sanity");
#if PCALL_ENABLE_NUMPY_CAPI
template <class T>
struct numpy_typecode {
  static constexpr int value =
      std::is_same_v<T, double>                                               ? NPY_DOUBLE
      : std::is_same_v<T, float>                                              ? NPY_FLOAT
      : std::is_same_v<T, int64_t>                                            ? NPY_INT64
      : std::is_same_v<T, bool>                                               ? NPY_BOOL
      : (std::is_integral_v<T> && !std::is_same_v<T, bool> && sizeof(T) == 4) ? NPY_INT32
                                                                              : -1;
};
#endif

/** Borrowed view: no ownership; valid only while the source memory stays alive.
 *  `writeable=true` lets Python mutate your buffer.
 */
template <class T>
struct ArrayView {
  T *data = nullptr;
  size_t size = 0;
  bool writeable = std::is_const<T>::value ? false : true;
};

template <class T>
inline ArrayView<T> view(std::vector<T> &v) {
  return {v.data(), v.size(), true};
}
template <class T>
inline ArrayView<const T> view(const std::vector<T> &v) {
  return {v.data(), v.size(), false};
}
// vector<bool> has no .data(); block zero-copy helpers for it.
template <>
inline pcall::ArrayView<bool> view(std::vector<bool> &) = delete;
template <>
inline pcall::ArrayView<const bool> view(const std::vector<bool> &) = delete;

#if PCALL_ENABLE_NUMPY_CAPI
template <class T>
struct ArrayViewND {
  T *data = nullptr;
  std::vector<size_t> shape; // C-order
  bool writeable = !std::is_const<T>::value;
  bool squeeze = false;
};

template <class T>
inline ArrayViewND<T> view_nd(std::vector<T> &v, std::vector<size_t> shp,
                              bool squeeze = false) {
  if (_prod(shp) != v.size()) {
    throw PyException("view_nd: size mismatch");
  }
  return {v.data(), shp, true, squeeze};
}
template <class T>
inline ArrayViewND<const T> view_nd(const std::vector<T> &v, std::vector<size_t> shp,
                                    bool squeeze = false) {
  if (_prod(shp) != v.size()) {
    throw PyException("view_nd: size mismatch");
  }
  return {v.data(), shp, false, squeeze};
}
#endif

/** Owned buffer: memory outlives the call; safe for Python to retain.
 *  Construct from a moved vector (zero extra copies).
 */
template <class T>
struct OwningArray {
  std::shared_ptr<std::vector<T>> holder;
  static OwningArray<T> from_vector(std::vector<T> &&v) {
    OwningArray<T> o;
    o.holder = std::make_shared<std::vector<T>>(
        std::move(v)); // takes ownership (no element copy)
    return o;
  }
};

// capsule cleanup for OwningArray
#if PCALL_ENABLE_NUMPY_CAPI
inline void _capsule_drop_vec(PyObject *capsule) {
  // steals back pointer and deletes it
  void *raw = PyCapsule_GetPointer(capsule, "pcall.owning_array");
  auto *p = static_cast<std::shared_ptr<void> *>(raw);
  delete p; // drops the shared_ptr, freeing vector if last owner
}
#endif

// ------------ C++ -> PyObject converters ------------

template <class T, class Enable = void>
struct ToPy;

// arithmetic
template <class T>
struct ToPy<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>> {
  static PyObject *convert(T v) { return PyLong_FromLongLong(static_cast<long long>(v)); }
};
template <>
struct ToPy<double> {
  static PyObject *convert(double v) { return PyFloat_FromDouble(v); }
};
template <>
struct ToPy<float> {
  static PyObject *convert(float v) { return PyFloat_FromDouble(static_cast<double>(v)); }
};
template <>
struct ToPy<bool> {
  static PyObject *convert(bool v) { return PyBool_FromLong(v ? 1 : 0); }
};

// strings
template <>
struct ToPy<std::string> {
  static PyObject *convert(const std::string &s) {
    return PyUnicode_FromString(s.c_str());
  }
};
template <>
struct ToPy<const char *> {
  static PyObject *convert(const char *s) { return PyUnicode_FromString(s); }
};

// std::vector -> NumPy ndarray via high-level numpy.array(list, dtype=...)
template <class T>
struct ToPy<std::vector<T>, std::enable_if_t<is_supported_vector_elem<T>::value>> {
  static PyObject *convert(const std::vector<T> &v) {
    GILGuard gil;
    // Build a Python list
    PyPtr list(PyList_New(static_cast<Py_ssize_t>(v.size())));
    if (!list) throw PyException("Failed to allocate Python list");
    for (Py_ssize_t i = 0; i < static_cast<Py_ssize_t>(v.size()); ++i) {
      PyObject *item = ToPy<T>::convert(v[static_cast<size_t>(i)]);
      if (!item) throw PyException("Failed to convert list element");
      // PyList_SET_ITEM steals reference
      PyList_SET_ITEM(list.get(), i, item);
    }
    // Call numpy.array(list, dtype=...)
    PyObject *np = PythonEnv::numpy_module(); // borrowed
    PyPtr array_func(PyObject_GetAttrString(np, "array"));
    if (!array_func) throw PyException("numpy.array not found");
    PyPtr args(PyTuple_New(1));
    if (!args) throw PyException("Failed to build arg tuple");
    Py_INCREF(list.get()); // because Tuple will steal
    PyTuple_SET_ITEM(args.get(), 0, list.get());
    PyPtr kwargs(PyDict_New());
    if (!kwargs) throw PyException("Failed to build kwargs");
    PyPtr dtype_str(PyUnicode_FromString(numpy_dtype_string<T>::value));
    if (!dtype_str) throw PyException("Failed to build dtype string");
    if (PyDict_SetItemString(kwargs.get(), "dtype", dtype_str.get()) != 0) {
      throw PyException("Failed to set dtype kwarg");
    }
    PyObject *out = PyObject_Call(array_func.get(), args.get(), kwargs.get());
    if (!out) throw PyException("numpy.array failed: " + fetch_python_error());
    return out; // new reference
  }
};

#if PCALL_ENABLE_NUMPY_CAPI
// Borrowed view -> NumPy ndarray (no copy). BEWARE lifetime!
template <class T>
struct ToPy<ArrayView<T>, void> {
  static PyObject *convert(const ArrayView<T> &v) {
    if (!v.data) throw PyException("ArrayView has null data");
    npy_intp dims[1] = {static_cast<npy_intp>(v.size)};
    using U = std::remove_const_t<T>;
    PyObject *arr =
        PyArray_SimpleNewFromData(1, dims, numpy_typecode<U>::value,
                                  const_cast<U *>(reinterpret_cast<const U *>(v.data)));
    if (!arr) throw PyException("NumPy view creation failed: " + fetch_python_error());
    if (!v.writeable) {
      PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(arr), NPY_ARRAY_WRITEABLE);
    }
    return arr; // new ref; base is null (borrowed)
  }
};

template <class T>
struct ToPy<ArrayViewND<T>, void> {
  static PyObject *convert(const ArrayViewND<T> &v) {
    if (!v.data) throw PyException("ArrayViewND has null data");
    std::vector<npy_intp> shp_np;
    if (v.squeeze) {
      if (v.shape.size() == 2) {
        if (v.shape[1] == 1) {
          shp_np.resize(1);
          shp_np[0] = static_cast<npy_intp>(v.shape[0]);
        } else {
          shp_np.resize(2);
          for (int i = 0; i < 2; i++)
            shp_np[i] = static_cast<npy_intp>(v.shape[i]);
        }
      } else {
        shp_np = _to_npy_shape(v.shape);
      }
    } else {
      shp_np = _to_npy_shape(v.shape);
    }
    using U = std::remove_const_t<T>;
    PyObject *arr = PyArray_SimpleNewFromData(
        static_cast<int>(shp_np.size()), shp_np.data(), numpy_typecode<U>::value,
        const_cast<U *>(reinterpret_cast<const U *>(v.data)));
    if (!arr) throw PyException("NumPy ND view creation failed: " + fetch_python_error());
    if (!v.writeable) {
      PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject *>(arr), NPY_ARRAY_WRITEABLE);
    }
    return arr;
  }
};

// Owned buffer -> NumPy ndarray (no copy) + base capsule carries lifetime.
template <class T>
struct ToPy<OwningArray<T>, void> {
  static PyObject *convert(const OwningArray<T> &o) {
    if (!o.holder || o.holder->empty()) {
      // empty is OK; still make an array of length 0
    }
    npy_intp dims[1] = {
        static_cast<npy_intp>(o.holder ? static_cast<npy_intp>(o.holder->size()) : 0)};
    PyObject *arr = PyArray_SimpleNewFromData(
        1, dims, numpy_typecode<T>::value,
        o.holder ? static_cast<void *>(o.holder->data()) : nullptr);
    if (!arr)
      throw PyException("NumPy owned view creation failed: " + fetch_python_error());

    // Create a capsule that owns a heap-allocated shared_ptr<void>, so Python holds the
    // memory.
    auto *heap_sp = new std::shared_ptr<void>(o.holder); // type-erased shared_ptr
    PyObject *cap = PyCapsule_New(heap_sp, "pcall.owning_array", &_capsule_drop_vec);
    if (!cap) {
      Py_DECREF(arr);
      delete heap_sp;
      throw PyException("Capsule creation failed: " + fetch_python_error());
    }
    // Set capsule as base object (steals ref to cap on success)
    if (PyArray_SetBaseObject(reinterpret_cast<PyArrayObject *>(arr), cap) != 0) {
      Py_DECREF(arr);
      Py_DECREF(cap);
      throw PyException("PyArray_SetBaseObject failed: " + fetch_python_error());
    }
    return arr;
  }
};

#else
// If NumPy C-API is disabled, make using zero-copy wrappers a compile error:
template <class T>
struct ToPy<ArrayView<T>, void> {
  static PyObject *convert(const ArrayView<T> &) {
    static_assert(!sizeof(T),
                  "Enable PCALL_ENABLE_NUMPY_CAPI=1 to use ArrayView (zero-copy).");
    return nullptr;
  }
};
template <class T>
struct ToPy<OwningArray<T>, void> {
  static PyObject *convert(const OwningArray<T> &) {
    static_assert(!sizeof(T),
                  "Enable PCALL_ENABLE_NUMPY_CAPI=1 to use OwningArray (zero-copy).");
    return nullptr;
  }
};
#endif

// ------------ PyObject -> C++ converters ------------

template <class T, class Enable = void>
struct FromPy;

// arithmetic
template <class T>
struct FromPy<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>> {
  static T convert(PyObject *o) {
    if (!PyLong_Check(o)) throw PyException("Expected int");
    long long v = PyLong_AsLongLong(o);
    if (PyErr_Occurred()) throw PyException(fetch_python_error());
    return static_cast<T>(v);
  }
};
template <>
struct FromPy<double> {
  static double convert(PyObject *o) {
    if (PyFloat_Check(o)) return PyFloat_AsDouble(o);
    if (PyLong_Check(o)) return static_cast<double>(PyLong_AsLongLong(o));
    throw PyException("Expected float");
  }
};
template <>
struct FromPy<float> {
  static float convert(PyObject *o) {
    return static_cast<float>(FromPy<double>::convert(o));
  }
};
template <>
struct FromPy<bool> {
  static bool convert(PyObject *o) {
    if (PyBool_Check(o)) return o == Py_True;
    if (PyLong_Check(o)) return PyLong_AsLongLong(o) != 0;
    throw PyException("Expected bool");
  }
};

// strings
template <>
struct FromPy<std::string> {
  static std::string convert(PyObject *o) {
    if (!PyUnicode_Check(o)) throw PyException("Expected str");
    const char *s = PyUnicode_AsUTF8(o);
    if (!s) throw PyException(fetch_python_error());
    return std::string(s);
  }
};

// buffers / sequences -> std::vector<T>
template <class T>
struct FromPy<std::vector<T>, std::enable_if_t<is_supported_vector_elem<T>::value>> {
  static std::vector<T> convert(PyObject *o) {
    auto get_contig = [](PyObject *obj, Py_buffer *view) -> bool {
      // Try to get a contiguous buffer (any ndim)
      if (PyObject_GetBuffer(obj, view, PyBUF_CONTIG_RO | PyBUF_FORMAT) == 0) return true;

      // If not contiguous, ask NumPy to ravel it
      PyErr_Clear();
      PyObject *np = nullptr;
      {
        try {
          np = PythonEnv::numpy_module();
        } catch (...) {
          np = nullptr;
        }
      }
      if (!np) return false;

      PyObject *ravel = PyObject_GetAttrString(np, "ravel");
      if (!ravel) {
        PyErr_Clear();
        return false;
      }

      PyObject *args = PyTuple_Pack(1, obj);
      PyObject *kwargs = Py_BuildValue("{s:s}", "order", "C");
      PyObject *flat = ravel ? PyObject_Call(ravel, args, kwargs) : nullptr;
      Py_XDECREF(ravel);
      Py_XDECREF(args);
      Py_XDECREF(kwargs);

      if (!flat) {
        PyErr_Clear();
        return false;
      }
      bool ok = (PyObject_GetBuffer(flat, view, PyBUF_CONTIG_RO | PyBUF_FORMAT) == 0);
      Py_DECREF(flat);
      return ok;
    };

    Py_buffer view;
    if (!get_contig(o, &view)) {
      // Fallback: sequence iteration
      if (!PySequence_Check(o)) throw PyException("Expected a NumPy array or a sequence");
      Py_ssize_t n = PySequence_Size(o);
      if (n < 0) throw PyException(fetch_python_error());
      std::vector<T> out;
      out.reserve(static_cast<size_t>(n));
      for (Py_ssize_t i = 0; i < n; ++i) {
        PyPtr item(PySequence_GetItem(o, i));
        if (!item) throw PyException(fetch_python_error());
        if constexpr (std::is_same_v<T, double>)
          out.push_back(FromPy<double>::convert(item.get()));
        else if constexpr (std::is_same_v<T, float>)
          out.push_back(static_cast<float>(FromPy<double>::convert(item.get())));
        else if constexpr (std::is_same_v<T, bool>)
          out.push_back(FromPy<bool>::convert(item.get()));
        else {
          long long v = FromPy<long long>::convert(item.get());
          out.push_back(static_cast<T>(v));
        }
      }
      return out;
    }

    // Contiguous flat copy (works for any ndim)
    const bool size_ok = (view.itemsize == sizeof(T));
    bool format_ok = true;
    if (view.format) {
      const char code = view.format[0];
      if constexpr (std::is_same_v<T, double>)
        format_ok = (code == 'd' || code == 'f');
      else if constexpr (std::is_same_v<T, float>)
        format_ok = (code == 'f' || code == 'd');
      else if constexpr (std::is_same_v<T, int64_t>)
        format_ok = (code == 'q' || code == 'l' || code == 'i');
      else if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int>)
        format_ok = (code == 'i' || code == 'l' || code == 'q');
      else if constexpr (std::is_same_v<T, bool>)
        format_ok = (code == '?' || code == 'b' || code == 'B' || code == 'i');
    }
    if (!size_ok) {
      PyBuffer_Release(&view);
      throw PyException("Itemsize mismatch");
    }

    const size_t n_elems = static_cast<size_t>(view.len) / sizeof(T);
    std::vector<T> out;
    out.resize(n_elems);
    if (format_ok) {
      std::memcpy(out.data(), view.buf, n_elems * sizeof(T));
    } else {
      // Slow path: convert elementwise if the buffer's format is incompatible
      const unsigned char *p = static_cast<const unsigned char *>(view.buf);
      for (size_t i = 0; i < n_elems; ++i) {
        // Reinterpret best-effort; real code could normalize more carefully
        out[i] = *(reinterpret_cast<const T *>(p + i * sizeof(T)));
      }
    }
    PyBuffer_Release(&view);
    return out;
  }
};

template <>
struct FromPy<std::vector<bool>> {
  static std::vector<bool> convert(PyObject *o) {
    std::vector<bool> out;

    // Fast path: buffer (NumPy array etc.) — treat any nonzero byte as true.
    if (PyObject_CheckBuffer(o)) {
      Py_buffer view;
      if (PyObject_GetBuffer(o, &view, PyBUF_CONTIG_RO) == 0) {
        const size_t n = static_cast<size_t>(view.len); // bytes
        out.reserve(n);
        const unsigned char *p = static_cast<const unsigned char *>(view.buf);
        for (size_t i = 0; i < n; ++i)
          out.push_back(p[i] != 0);
        PyBuffer_Release(&view);
        return out;
      }
      // if buffer failed, fall through to sequence
    }

    // Generic sequence fallback (list/tuple/etc.)
    if (!PySequence_Check(o)) {
      throw PyException("Expected a NumPy array or a sequence for vector<bool>");
    }
    Py_ssize_t n = PySequence_Size(o);
    if (n < 0) throw PyException(fetch_python_error());
    out.reserve(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
      PyPtr item(PySequence_GetItem(o, i));
      if (!item) throw PyException(fetch_python_error());
      out.push_back(FromPy<bool>::convert(item.get()));
    }
    return out;
  }
};

// ------------ tuple builder ------------

template <size_t I = 0, class... Ts>
inline std::enable_if_t<I == sizeof...(Ts), void>
tuple_set_items(PyObject *, const std::tuple<Ts...> &) {}

template <size_t I = 0, class... Ts>
    inline std::enable_if_t <
    I<sizeof...(Ts), void> tuple_set_items(PyObject *tup, const std::tuple<Ts...> &tpl) {
  using Elem = std::decay_t<std::tuple_element_t<I, std::tuple<Ts...>>>;
  PyObject *obj = ToPy<Elem>::convert(std::get<I>(tpl));
  if (!obj) throw PyException("Argument conversion failed");
  // steals reference
  PyTuple_SET_ITEM(tup, static_cast<Py_ssize_t>(I), obj);
  tuple_set_items<I + 1, Ts...>(tup, tpl);
}

// ------------ main wrapper: PyFunction ------------

class PyFunction {
  PyPtr module_;
  PyPtr callable_;

 public:
  PyFunction(const std::string &module_name_or_path, const std::string &function_name,
             const std::vector<std::string> &extra_sys_path,
             const std::string &import_as_alias) {
    GILGuard gil;

    module_ = PyPtr(_import_module_by_name_or_file(module_name_or_path, import_as_alias));
    if (!module_)
      throw PyException("Failed to import '" + module_name_or_path +
                        "': " + fetch_python_error());

    callable_ = PyPtr(PyObject_GetAttrString(module_.get(), function_name.c_str()));
    if (!callable_ || !PyCallable_Check(callable_.get()))
      throw PyException("Function '" + function_name + "' not found/callable");
  }

  // convenience: alias optional
  PyFunction(const std::string &module_name_or_path, const std::string &function_name,
             const std::vector<std::string> &extra_sys_path = {})
      : PyFunction(module_name_or_path, function_name, extra_sys_path, /*alias*/ "") {}

  template <class R = void, class... Args>
  R call(Args &&...args) const {
    GILGuard gil;
    constexpr size_t N = sizeof...(Args);
    PyPtr args_tuple(PyTuple_New(static_cast<Py_ssize_t>(N)));
    if (!args_tuple) throw PyException("Failed to create args tuple");
    auto packed = std::make_tuple(std::forward<Args>(args)...);
    tuple_set_items(args_tuple.get(), packed);

    PyPtr result(PyObject_CallObject(callable_.get(), args_tuple.get()));
    if (!result) throw PyException("Python call failed: " + fetch_python_error());

    if constexpr (std::is_same_v<R, void>) {
      return;
    } else {
      return FromPy<R>::convert(result.get());
    }
  }

  template <class... Args>
  void call_void(Args &&...args) const {
    (void)call<void>(std::forward<Args>(args)...);
  }

  template <class... Args>
  auto operator()(Args &&...args) const {
    return call<PyObject *>(std::forward<Args>(args)...);
  } // raw PyObject* if you want
};

class PyObjectRef {
  PyPtr obj_;

 public:
  PyObjectRef() = default;
  explicit PyObjectRef(PyObject *o) : obj_(o) {}
  PyObject *raw() const { return (obj_ ? obj_.get() : nullptr); }
  explicit operator bool() const { return static_cast<bool>(obj_); }

  // Call method by name: obj.method(args...) -> R
  template <class R = void, class... Args>
  R call_method(const char *name, Args &&...args) const {
    if (!obj_) throw PyException("Attempting to call an uninitialized python object");
    GILGuard gil;

#if PCALL_HAVE_VECTORCALL_METHOD
    // Fast path: avoid creating a bound method object
    PyPtr name_str(PyUnicode_FromString(name));
    if (!name_str) throw PyException("Failed to build method name");
    constexpr size_t N = sizeof...(Args);
    PyObject *argv[1 + N] = {
        obj_.get(), ToPy<std::decay_t<Args>>::convert(std::forward<Args>(args))...};

    struct Cleaner {
      PyObject **a;
      size_t n;
      ~Cleaner() {
        for (size_t i = 1; i < n; ++i)
          Py_XDECREF(a[i]);
      }
    } cleaner{argv, 1 + N};

    size_t nargsf = (1 + N) | PY_VECTORCALL_ARGUMENTS_OFFSET;
    PyObject *res = PyObject_VectorcallMethod(name_str.get(), argv, nargsf, nullptr);
    if (!res) throw PyException("Method call failed: " + fetch_python_error());
    PyPtr result(res);
    if constexpr (std::is_same_v<R, void>)
      return;
    else
      return FromPy<R>::convert(result.get());
#else
    // Fallback: get bound method then call
    PyPtr meth(PyObject_GetAttrString(obj_.get(), name));
    if (!meth || !PyCallable_Check(meth.get()))
      throw PyException(std::string("Attribute '") + name + "' not callable");

    constexpr size_t N = sizeof...(Args);
    PyObject *argv[N] = {ToPy<std::decay_t<Args>>::convert(std::forward<Args>(args))...};
    struct Cleaner {
      PyObject **a;
      size_t n;
      ~Cleaner() {
        for (size_t i = 0; i < n; ++i)
          Py_XDECREF(a[i]);
      }
    } cleaner{argv, N};

#if PCALL_HAVE_VECTORCALL
    if (PyVectorcall_Function(meth.get())) {
      PyObject *res = PyObject_Vectorcall(meth.get(), argv, N, nullptr);
      if (!res) throw PyException("Method call failed: " + fetch_python_error());
      PyPtr result(res);
      if constexpr (std::is_same_v<R, void>)
        return;
      else
        return FromPy<R>::convert(result.get());
    }
#endif
    PyPtr tup(PyTuple_New(static_cast<Py_ssize_t>(N)));
    if (!tup) throw PyException("Failed to create args tuple");
    for (size_t i = 0; i < N; ++i) {
      Py_INCREF(argv[i]);
      PyTuple_SET_ITEM(tup.get(), static_cast<Py_ssize_t>(i), argv[i]);
    }
    PyPtr out(PyObject_CallObject(meth.get(), tup.get()));
    if (!out) throw PyException("Method call failed: " + fetch_python_error());
    if constexpr (std::is_same_v<R, void>)
      return;
    else
      return FromPy<R>::convert(out.get());
#endif
  }

  // Test for existence
  bool exists(const char *name) const {
    if (!obj_) return false;
    GILGuard gil;
    PyPtr attr(PyObject_GetAttrString(obj_.get(), name));
    PyErr_Clear();
    return static_cast<bool>(attr);
  }

  // Attribute get / set
  template <class T>
  T get_attr(const char *name) const {
    if (!obj_)
      throw PyException("Attempting to get_attr on an uninitialized python object");
    GILGuard gil;
    PyPtr attr(PyObject_GetAttrString(obj_.get(), name));
    if (!attr) throw PyException(std::string("No such attribute: ") + name);
    return FromPy<T>::convert(attr.get());
  }
  template <class T>
  void set_attr(const char *name, const T &value) const {
    if (!obj_)
      throw PyException("Attempting to set_attr on an uninitialized python object");
    GILGuard gil;
    PyPtr v(ToPy<std::decay_t<T>>::convert(value));
    if (!v) throw PyException("Attribute conversion failed");
    if (PyObject_SetAttrString(obj_.get(), name, v.get()) != 0)
      throw PyException("Failed to set attribute: " + fetch_python_error());
  }
  template <class T>
  T get_or_set_attr(const char *name, const T &value) const {
    if (!obj_)
      throw PyException(
          "Attempting to get_or_set_attr on an uninitialized python object");
    GILGuard gil;
    PyPtr attr(PyObject_GetAttrString(obj_.get(), name));
    if (!attr) {
      set_attr(name, value);
      return value;
    }
    return FromPy<T>::convert(attr.get());
  }
};

class PyClass {
  PyPtr module_;
  PyPtr cls_;

 public:
  PyClass(const std::string &module_name_or_path, const std::string &class_name,
          const std::vector<std::string> &extra_sys_path = {},
          const std::string &import_as_alias = "") {
    GILGuard gil;

    module_ = PyPtr(_import_module_by_name_or_file(module_name_or_path, import_as_alias));
    if (!module_) throw PyException("Failed to import module: " + fetch_python_error());

    cls_ = PyPtr(PyObject_GetAttrString(module_.get(), class_name.c_str()));
    if (!cls_) throw PyException("Class not found: " + class_name);
    if (!PyCallable_Check(cls_.get()))
      throw PyException("Object is not a class: " + class_name);
  }

  // Construct instance: returns a PyObjectRef you can use to call methods
  template <class... Args>
  PyObject *new_instance(Args &&...args) const {
    GILGuard gil;
    constexpr size_t N = sizeof...(Args);
    PyObject *argv[N] = {ToPy<std::decay_t<Args>>::convert(std::forward<Args>(args))...};
    struct Cleaner {
      PyObject **a;
      size_t n;
      ~Cleaner() {
        for (size_t i = 0; i < n; ++i)
          Py_XDECREF(a[i]);
      }
    } cleaner{argv, N};

#if PCALL_HAVE_VECTORCALL
    if (PyVectorcall_Function(cls_.get())) {
      PyObject *inst = PyObject_Vectorcall(cls_.get(), argv, N, nullptr);
      if (!inst) throw PyException("Class construction failed: " + fetch_python_error());
      // return PyObjectRef(inst);
      return inst;
    }
#endif
    PyPtr tup(PyTuple_New(static_cast<Py_ssize_t>(N)));
    if (!tup) throw PyException("Failed to create args tuple");
    if constexpr (N > 0) {
      for (size_t i = 0; i < N; ++i) {
        Py_INCREF(argv[i]);
        PyTuple_SET_ITEM(tup.get(), static_cast<Py_ssize_t>(i), argv[i]);
      }
    }
    PyObject *inst = PyObject_CallObject(cls_.get(), tup.get());
    if (!inst) throw PyException("Class construction failed: " + fetch_python_error());
    // return PyObjectRef(inst);
    return inst;
  }
  template <class... Args>
  PyObjectRef new_object(Args &&...args) const {
    return PyObjectRef(new_instance(std::forward<Args>(args)...));
  }
};

struct MethodArity {
  int min_positional = 0;   // required positional (after binding; no 'self')
  int max_positional = 0;   // total positional (after binding); -1 if unbounded (*args)
  int kwonly_required = 0;  // required keyword-only
  int kwonly_total = 0;     // total keyword-only
  bool has_varargs = false; // *args present
  bool has_varkw = false;   // **kwargs present
};

// Internal: use inspect.signature to compute arity
inline MethodArity _arity_via_inspect(PyObject *callable) {
  MethodArity a;
  PyPtr inspect(PyImport_ImportModule("inspect"));
  if (!inspect) throw PyException("import inspect failed: " + fetch_python_error());

  PyPtr signature(PyObject_GetAttrString(inspect.get(), "signature"));
  if (!signature) throw PyException("inspect.signature missing");

  PyPtr sig(PyObject_CallFunctionObjArgs(signature.get(), callable, nullptr));
  if (!sig)
    throw PyException("inspect.signature(callable) failed: " + fetch_python_error());

  PyPtr params(PyObject_GetAttrString(sig.get(), "parameters")); // mappingproxy
  if (!params) throw PyException("signature.parameters missing");

  PyPtr values(
      PyObject_CallMethod(params.get(), "values", nullptr)); // values view (iterable)
  if (!values) throw PyException("parameters.values() failed");

  PyPtr empty(
      PyObject_GetAttrString(inspect.get(), "_empty")); // sentinel for “no default”
  if (!empty) throw PyException("inspect._empty missing");

  PyPtr iter(PyObject_GetIter(values.get()));
  if (!iter) throw PyException("iter(parameters) failed");

  a.min_positional = 0;
  int pos_total = 0;
  int max_args = 1024; // totally arbitrary maximum number of arguments
  int iarg = 0;
  while (iarg < max_args) {
    PyPtr p(PyIter_Next(iter.get()));
    if (!p) break; // end
    iarg++;
    // kind: 0=POSITIONAL_ONLY, 1=POSITIONAL_OR_KEYWORD, 2=VAR_POSITIONAL, 3=KEYWORD_ONLY,
    // 4=VAR_KEYWORD
    PyPtr kind(PyObject_GetAttrString(p.get(), "kind"));
    if (!kind) throw PyException("Parameter.kind missing");
    long k = PyLong_AsLong(kind.get());
    if (PyErr_Occurred()) throw PyException(fetch_python_error());

    // default present?
    PyPtr def(PyObject_GetAttrString(p.get(), "default"));
    if (!def) throw PyException("Parameter.default missing");
    bool has_default = (def.get() != empty.get());

    switch (k) {
    case 0: // POSITIONAL_ONLY
    case 1: // POSITIONAL_OR_KEYWORD
      ++pos_total;
      if (!has_default) ++a.min_positional;
      break;
    case 2: // VAR_POSITIONAL (*args)
      a.has_varargs = true;
      break;
    case 3: // KEYWORD_ONLY
      ++a.kwonly_total;
      if (!has_default) ++a.kwonly_required;
      break;
    case 4: // VAR_KEYWORD (**kwargs)
      a.has_varkw = true;
      break;
    default:
      break;
    }
  }
  PARTHENON_REQUIRE(iarg < max_args,
                    "Python function appears to have a wild-a** number of args.  "
                    "Something unexpected is likely happening.");

  a.max_positional = a.has_varargs ? -1 : pos_total;
  return a;
}

// Fallback: quick check for builtins implemented as PyCFunction
inline bool _arity_via_cfunc(PyObject *callable, MethodArity &out) {
  if (!PyCFunction_Check(callable)) return false;
  int flags = PyCFunction_GET_FLAGS(callable);
  // Note: we're inspecting the *bound* method if attr came from an instance.
  if (flags & METH_NOARGS) {
    out = {0, 0, 0, 0, false, false};
  } else if (flags & METH_O) {
    out = {1, 1, 0, 0, false, false};
  } else {
    // METH_VARARGS / FASTCALL etc. → variable arity; treat as unbounded
    out = {0,
           -1,
           0,
           0,
           bool(flags & (METH_VARARGS | METH_FASTCALL)),
           bool(flags & METH_KEYWORDS)};
  }
  return true;
}

// Public API: given an object and method name, determine arity (after binding)
inline MethodArity get_method_arity(const PyObjectRef &obj, const char *method_name) {
  GILGuard gil;

  // Fetch attribute; when taken from an instance, this is a *bound* method already.
  PyPtr attr(PyObject_GetAttrString(obj.raw(), method_name));
  if (!attr) throw PyException(std::string("No such attribute: ") + method_name);
  if (!PyCallable_Check(attr.get()))
    throw PyException(std::string("Attribute '") + method_name + "' is not callable");

  // Prefer inspect (handles Python functions, many builtins via __text_signature__)
  try {
    return _arity_via_inspect(attr.get());
  } catch (...) {
    // Fall back to PyCFunction flags if possible
    MethodArity a;
    if (_arity_via_cfunc(attr.get(), a)) return a;
    // Unknown — conservatively report “variable”
    MethodArity unk;
    unk.min_positional = 0;
    unk.max_positional = -1;
    unk.has_varargs = true;
    return unk;
  }
}

// convenience free function
template <class R = void, class... Args>
R call(const std::string &module_name, const std::string &function_name, Args &&...args) {
  PyFunction f(module_name, function_name);
  return f.template call<R>(std::forward<Args>(args)...);
}

} // namespace pcall

#endif // RIOT_UTILS_PY_CALLER_HPP_
