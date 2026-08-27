//========================================================================================
// (C) (or copyright) 2023-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef RIOT_UTILS_TYPE_MAPS_HPP_
#define RIOT_UTILS_TYPE_MAPS_HPP_
// This file was made in part with generative AI.

#include <type_traits>

namespace TypeMaps {

// Pair Key->Value
template <typename Key, typename Value>
struct TypePair {
  using key = Key;
  using value = Value;
};

// Compile-time lookup (linear, keys assumed unique)
template <typename Key, typename Default, typename... Pairs>
struct lookup {
  using type = Default;
};

template <typename Key, typename Default, typename Pair, typename... Rest>
struct lookup<Key, Default, Pair, Rest...> {
  using type = std::conditional_t<std::is_same<typename Pair::key, Key>::value,
                                  typename Pair::value,
                                  typename lookup<Key, Default, Rest...>::type>;
};

// Compile-time lookup (linear, keys assumed unique)
template <typename Value, typename Default, typename... Pairs>
struct invert {
  using type = Default;
};

template <typename Value, typename Default, typename Pair, typename... Rest>
struct invert<Value, Default, Pair, Rest...> {
  using type = std::conditional_t<std::is_same<typename Pair::value, Value>::value,
                                  typename Pair::key,
                                  typename invert<Value, Default, Rest...>::type>;
};

// TypeMap holds the list; exposes a per-key result with ::type and ::exists
template <typename... Pairs>
struct GenericTypeMap {
  template <typename Key>
  struct at {
    using type = typename lookup<Key, void, Pairs...>::type;
    static constexpr bool exists = !std::is_same<type, void>::value;
  };
  template <typename Value>
  struct inv {
    using type = typename invert<Value, void, Pairs...>::type;
  };

  static constexpr int size() { return sizeof...(Pairs); }
  template <typename Key>
  static constexpr bool contains() {
    return at<Key>::exists;
  }
  template <template <typename> class F, typename... Args>
  static void for_each_key(Args &&...args) {
    (F<typename Pairs::key>::apply(std::forward<Args>(args)...), ...);
  }
  template <template <typename> class F, typename... Args>
  static void for_each_value(Args &&...args) {
    (F<typename Pairs::value>::apply(std::forward<Args>(args)...), ...);
  }
  template <template <typename, typename> class F, typename... Args>
  static void for_each_key_value(Args &&...args) {
    (F<typename Pairs::key, typename Pairs::value>::apply(std::forward<Args>(args)...),
     ...);
  }
};

template <typename T, typename GenericTypeMap_t>
struct TypeMap : public GenericTypeMap_t {
  using type = typename GenericTypeMap_t::template at<T>::type;
  using inverse = typename GenericTypeMap_t::template inv<T>::type;
  static constexpr bool contains = GenericTypeMap_t::template contains<T>();
  static constexpr int size() { return GenericTypeMap_t::size(); }
  template <template <typename> class F, typename... Args>
  static void for_each_key(Args &&...args) {
    GenericTypeMap_t::template for_each_key<F, Args...>(std::forward<Args>(args)...);
  }
  template <template <typename> class F, typename... Args>
  static void for_each_value(Args &&...args) {
    GenericTypeMap_t::template for_each_value<F, Args...>(std::forward<Args>(args)...);
  }
  template <template <typename, typename> class F, typename... Args>
  static void for_each_key_value(Args &&...args) {
    GenericTypeMap_t::template for_each_key_value<F, Args...>(
        std::forward<Args>(args)...);
  }
};

} // namespace TypeMaps
#endif // RIOT_UTILS_TYPE_MAPS_HPP_
