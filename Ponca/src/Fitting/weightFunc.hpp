/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

template <class DataPoint, class WeightKernel, class _NeighborhoodFrame>
typename DistWeightFunc<DataPoint, WeightKernel, _NeighborhoodFrame>::WeightReturnType DistWeightFunc<
    DataPoint, WeightKernel, _NeighborhoodFrame>::operator()(const DataPoint& _q) const
{
    const auto lq = NeighborhoodFrame::convertToLocalBasis(_q.pos());
    const auto d  = NeighborhoodFrame::localDistance(lq);

    if (isCompact) // compile-time branching
        return {(d <= 1.) ? m_wk.f(d) : Scalar(0.), lq};
    else
        return {m_wk.f(d), lq};
}

template <class DataPoint, class WeightKernel, class _NeighborhoodFrame>
typename DistWeightFunc<DataPoint, WeightKernel, _NeighborhoodFrame>::VectorType DistWeightFunc<
    DataPoint, WeightKernel, _NeighborhoodFrame>::spacedw(const VectorType& _q, const DataPoint&) const
{
    static_assert(WeightKernel::isDValid, "First order derivatives are required");
    static_assert(NeighborhoodFrame::hasDistanceSpaceDerivative, "First order distance derivatives are required");

    VectorType result = VectorType::Zero();
    const auto lq     = NeighborhoodFrame::convertToLocalBasis(_q);
    const auto d      = NeighborhoodFrame::localDistance(lq);
    // isCompact is tested at compile-time
    if ((!isCompact) || (d <= 1. && d != Scalar(0.)))
        result = m_wk.df(d) * NeighborhoodFrame::localDistancedw(lq, d);
    return result;
}

template <class DataPoint, class WeightKernel, class _NeighborhoodFrame>
typename DistWeightFunc<DataPoint, WeightKernel, _NeighborhoodFrame>::MatrixType DistWeightFunc<
    DataPoint, WeightKernel, _NeighborhoodFrame>::spaced2w(const VectorType& _q, const DataPoint&) const
{
    static_assert(WeightKernel::isDDValid, "Second order derivatives are required");
    static_assert(NeighborhoodFrame::hasDistanceSpaceDerivative, "First order distance derivatives are required");
    MatrixType result = MatrixType::Zero();

    const auto lq = NeighborhoodFrame::convertToLocalBasis(_q);
    const auto d  = NeighborhoodFrame::localDistance(lq);
    if ((!isCompact) || (d <= 1. && d != Scalar(0.)))
    {
        const auto g = NeighborhoodFrame::localDistancedw(lq, d);
        const auto h = NeighborhoodFrame::localDistanced2w(lq, g, d);
        result       = m_wk.ddf(d) * g * g.transpose() + m_wk.df(d) * h;
    }
    return result;
}

template <class DataPoint, class WeightKernel, class _NeighborhoodFrame>
typename DistWeightFunc<DataPoint, WeightKernel, _NeighborhoodFrame>::Scalar DistWeightFunc<
    DataPoint, WeightKernel, _NeighborhoodFrame>::scaledw(const VectorType& _q, const DataPoint&) const
{
    static_assert(WeightKernel::isDValid, "First order derivatives are required");
    static_assert(NeighborhoodFrame::hasDistanceScaleDerivative, "First order distance derivatives are required");

    const auto lq = NeighborhoodFrame::convertToLocalBasis(_q);
    const auto d  = NeighborhoodFrame::localDistance(lq);

    Scalar result = 0.;
    if ((!isCompact) || (d <= 1. && d != Scalar(0.)))
    {
        result = NeighborhoodFrame::localDistanceds(lq, d) * m_wk.df(d);
    }

    return result;
}

template <class DataPoint, class WeightKernel, class _NeighborhoodFrame>
typename DistWeightFunc<DataPoint, WeightKernel, _NeighborhoodFrame>::Scalar DistWeightFunc<
    DataPoint, WeightKernel, _NeighborhoodFrame>::scaled2w(const VectorType& _q, const DataPoint&) const
{
    static_assert(WeightKernel::isDDValid, "Second order derivatives are required");
    static_assert(NeighborhoodFrame::hasDistanceScaleDerivative, "Second order distance derivatives are required");

    const auto lq = NeighborhoodFrame::convertToLocalBasis(_q);
    const auto d  = NeighborhoodFrame::localDistance(lq);

    Scalar result = 0.;

    if ((!isCompact) || (d <= 1. && d != Scalar(0.)))
    {
        const auto g = NeighborhoodFrame::localDistanceds(lq, d);
        const auto h = NeighborhoodFrame::localDistanced2s(lq, g, d);
        result       = m_wk.ddf(d) * g * g + m_wk.df(d) * h;
    }
    return result;
}

template <class DataPoint, class WeightKernel, class _NeighborhoodFrame>
typename DistWeightFunc<DataPoint, WeightKernel, _NeighborhoodFrame>::VectorType DistWeightFunc<
    DataPoint, WeightKernel, _NeighborhoodFrame>::scaleSpaced2w(const VectorType& _q, const DataPoint&) const
{
    static_assert(WeightKernel::isDValid, "First order derivatives are required");
    static_assert(NeighborhoodFrame::hasDistanceSpaceDerivative && NeighborhoodFrame::hasDistanceScaleDerivative,
                  "Space/Scale derivative are required");

    const auto lq = NeighborhoodFrame::convertToLocalBasis(_q);
    const auto d  = NeighborhoodFrame::localDistance(lq);

    VectorType result = VectorType::Zero();
    if ((!isCompact) || (d <= 1. && d != Scalar(0.)))
    {
        const auto gt  = NeighborhoodFrame::localDistanceds(lq, d);
        const auto gx  = NeighborhoodFrame::localDistancedw(lq, d);
        const auto gtx = NeighborhoodFrame::localDistancedsdw(lq, d);
        result         = m_wk.ddf(d) * gt * gx + m_wk.df(d) * gtx;
    }

    return result;
}

