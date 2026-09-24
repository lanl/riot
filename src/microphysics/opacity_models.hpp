
//========================================================================================
// (C) (or copyright) 2024-2026. Triad National Security, LLC. All rights reserved.
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
#ifndef MICROPHYSICS_OPACITY_MODELS_HPP_
#define MICROPHYSICS_OPACITY_MODELS_HPP_
// This file was made in part with generative AI.

// Singularity-opac includes
#include <singularity-opac/photons/mean_opacity_photons.hpp>
#include <singularity-opac/photons/mean_s_opacity_photons.hpp>
#include <singularity-opac/photons/opac_photons.hpp>
#include <singularity-opac/photons/s_opac_photons.hpp>

namespace RiotOpacity {

//----------------------------------------------------------------------------------------
// Reduced absorption variant
using OpacA = singularity::photons::impl::Variant<singularity::photons::Gray,
                                                  singularity::photons::PowerLaw>;
using MeanOpacA = singularity::photons::MeanOpacityBase;

//----------------------------------------------------------------------------------------
// Reduced scattering variant
using OpacS = singularity::photons::impl::S_Variant<singularity::photons::GrayS,
                                                    singularity::photons::ThomsonS>;
using MeanOpacS = singularity::photons::MeanSOpacityBase;

} // namespace RiotOpacity

#endif // MICROPHYSICS_OPACITY_MODELS_HPP_
