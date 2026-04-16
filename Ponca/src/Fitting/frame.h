/*
 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#pragma once

#include "defines.h"

namespace Ponca
{

// Function that accepts DataPoint instead
#define PONCA_SPACE_FRAME_FUNCTIONS                                                                           \
    PONCA_MULTIARCH VectorType convertToLocalBasis(const DataPoint& _q, bool _isPositionVector = true) const  \
    {                                                                                                         \
        return convertToLocalBasis(_q.pos(), _isPositionVector);                                              \
    }                                                                                                         \
    PONCA_MULTIARCH VectorType convertToGlobalBasis(const DataPoint& _q, bool _isPositionVector = true) const \
    {                                                                                                         \
        return convertToGlobalBasis(_q.pos(), _isPositionVector);                                             \
    }                                                                                                         \
    PONCA_MULTIARCH void setCenter(const DataPoint& _q) { setCenter(_q.pos()); }

// Function without additionnal information for derivatives
#define PONCA_SPACE_FRAME_DERIVATIVES                                        \
    PONCA_MULTIARCH VectorType localDistancedw(const VectorType& _p) const   \
    {                                                                        \
        return localDistancedw(_p, localDistance(_p));                       \
    }                                                                        \
    PONCA_MULTIARCH MatrixType localDistanced2w(const VectorType& _p) const  \
    {                                                                        \
        Scalar d     = localDistance(_p);                                    \
        VectorType g = localDistancedw(_p, d);                               \
        return localDistanced2w(_p, g, d);                                   \
    }                                                                        \
    PONCA_MULTIARCH ScaleVector localDistanced2s(const VectorType& _p) const \
    {                                                                        \
        return localDistanceds(_p, localDistance(_p));                       \
    }                                                                        \
    PONCA_MULTIARCH ScaleMatrix localDistanced2s(const VectorType& _p)       \
    {                                                                        \
        Scalar d      = localDistance(_p);                                   \
        ScaleVector g = localDistancedw(_p, d);                              \
        return localDistanced2s(_p, g, d);                                   \
    }                                                                        \
    PONCA_MULTIARCH ScaleMatrix localDistancedsdw(const VectorType& _p)      \
    {                                                                        \
        Scalar d = localDistance(_p);                                        \
        return localDistancedsdw(_p, d);                                     \
    }

    template <typename _DataPoint>
    struct GlobalNeighborhoodFrame
    {
    public:
        using DataPoint  = _DataPoint;
        using Scalar     = _DataPoint::Scalar;
        using VectorType = _DataPoint::VectorType;
        using MatrixType = _DataPoint::MatrixType;
        // For non isotropic, this should be VectorType
        using ScaleVector = Scalar;
        using ScaleMatrix = Scalar;

        GlobalNeighborhoodFrame() {}
        GlobalNeighborhoodFrame(const DataPoint&, Scalar = 0.) {}
        GlobalNeighborhoodFrame(const VectorType&, Scalar = 0.) {}

        /// \brief Flag indicating that this class does not modify the coordinates when passing from global to local
        static constexpr bool hasLocalFrame = false;
        /// \brief Flag indicating that this class behaves the same in every directions
        static constexpr bool isIsotropic = true;
        /// \brief Flag indicating that this provides space derivative of distance
        static constexpr bool hasDistanceSpaceDerivative = true;
        /// \brief Flag indicating that this provides scale derivative of distance
        static constexpr bool hasDistanceScaleDerivative = true;

        PONCA_MULTIARCH VectorType convertToLocalBasis(const VectorType& _q, bool _isPositionVector = true) const
        {
            return _q;
        }

        PONCA_MULTIARCH VectorType convertToGlobalBasis(const VectorType& _q, bool _isPositionVector = true) const
        {
            return _q;
        }

        PONCA_MULTIARCH const VectorType& center() const { return VectorType{}; }
        PONCA_MULTIARCH void setCenter(const VectorType& _q) {}
        PONCA_MULTIARCH void rescale(Scalar s) {}

        PONCA_MULTIARCH ScaleVector scale() const { return 1.; }
        PONCA_MULTIARCH ScaleVector iScale() const { return 1.; }

        // Distance from the origin for a point in local coordinates
        PONCA_MULTIARCH Scalar localDistance(const VectorType& _q) const { return _q.norm(); }

        // Gradient of distance from origin in space for a point in local coordinates
        // This version also expect the distance computed by localDistance, which can help reduce computationnal cost.
        PONCA_MULTIARCH VectorType localDistancedw(const VectorType& _q, Scalar _d) const { return _q / _d; }
        PONCA_MULTIARCH MatrixType localDistanced2w(const VectorType& _q, const VectorType& _g, Scalar _d) const
        {
            Scalar hScale = -1. / (_d * _d * _d);

            MatrixType H = hScale * (_q * _q.transpose());
            H.diagonal().array() += _g;

            return H;
        }

        PONCA_MULTIARCH ScaleVector localDistanceds(const VectorType& _q, Scalar _d) const { return 0.; }
        PONCA_MULTIARCH ScaleMatrix localDistanced2s(const VectorType& _q, const ScaleVector& _g, Scalar _d) const
        {
            return 0.;
        }

        PONCA_MULTIARCH ScaleVector localDistancedsdw(const VectorType& _q, Scalar _d) const { return 0.; }

        PONCA_SPACE_FRAME_FUNCTIONS
        PONCA_SPACE_FRAME_DERIVATIVES
    };

    template <typename _Base>
    struct FrameWithNormal : public _Base
    {
        using Base       = _Base;
        using DataPoint   = typename Base::DataPoint;
        using Scalar      = typename Base::Scalar;
        using VectorType  = typename Base::VectorType;
        using MatrixType  = typename Base::MatrixType;
        using ScaleVector = typename Base::ScaleVector;
        using ScaleMatrix = typename Base::ScaleMatrix;


        /// \brief Flag indicating that this class does not modify the coordinates when passing from global to local
        static constexpr bool hasLocalFrame = Base::hasLocalFrame;
        /// \brief Flag indicating that this class behaves the same in every directions
        static constexpr bool isIsotropic = Base::isIsotropic;
        /// \brief Flag indicating that this provides space derivative of distance
        static constexpr bool hasDistanceSpaceDerivative = Base::hasDistanceSpaceDerivative;
        /// \brief Flag indicating that this provides scale derivative of distance
        static constexpr bool hasDistanceScaleDerivative = Base::hasDistanceScaleDerivative;

        FrameWithNormal() : Base(), m_normal(VectorType::Zero()) {}
        FrameWithNormal(const VectorType& _c, const Scalar _t = 1.) : Base(_c, _t), m_normal(VectorType::Zero()) {}
        FrameWithNormal(const VectorType& _c, const Scalar _t, const VectorType& _n)
            : Base(_c, _t), m_normal(_n / _n.norm())
        {
        }
        FrameWithNormal(const DataPoint& _p, const Scalar _t = 1.) : FrameWithNormal(_p.pos(), _t, _p.normal()) {}

        PONCA_MULTIARCH void setNormal(const VectorType& _n, bool isNormalized = false)
        {
            if (isNormalized)
                m_normal = _n;
            else
                m_normal = _n / _n.norm();
        }

        PONCA_MULTIARCH void setCenter(const DataPoint& _q) const
        {
            Base::setCenter(_q.pos());
            setNormal(_q.normal());
        }

        PONCA_MULTIARCH const VectorType& normal() const { return m_normal; }

    protected:
        typename Base::VectorType m_normal;
    };

    template <typename _DataPoint>
    struct CenteredNeighborhoodFrame
    {
    public:
        using DataPoint   = _DataPoint;
        using Scalar      = _DataPoint::Scalar;
        using VectorType  = _DataPoint::VectorType;
        using MatrixType  = _DataPoint::MatrixType;
        using ScaleVector = Scalar;
        using ScaleMatrix = Scalar;

        /// \brief Flag indicating that this class does not modify the coordinates when passing from global to local
        static constexpr bool hasLocalFrame = true;
        /// \brief Flag indicating that this class behaves the same in every directions
        static constexpr bool isIsotropic = true;
        /// \brief Flag indicating that this provides space derivative of distance
        static constexpr bool hasDistanceSpaceDerivative = true;
        /// \brief Flag indicating that this provides scale derivative of distance
        static constexpr bool hasDistanceScaleDerivative = true;

        CenteredNeighborhoodFrame() : m_center(VectorType::Zero()), m_scale(1.), m_iScale(1.) {}

        CenteredNeighborhoodFrame(const VectorType& _q, Scalar _t = 1.) : CenteredNeighborhoodFrame()
        {
            setCenter(_q);
            rescale(_t);
        }

        CenteredNeighborhoodFrame(const DataPoint& _p, Scalar _t = 1.) : CenteredNeighborhoodFrame(_p.pos(), _t) {}

        PONCA_MULTIARCH VectorType convertToLocalBasis(const VectorType& _q, bool _isPositionVector = true) const
        {
            if (_isPositionVector)
                return (_q - m_center) * m_iScale;
            return _q;
        }

        PONCA_MULTIARCH VectorType convertToGlobalBasis(const VectorType& _q, bool _isPositionVector = true) const
        {
            if (_isPositionVector)
                return _q * m_scale + m_center;
            return _q;
        }

        PONCA_MULTIARCH const VectorType& center() const { return m_center; }
        PONCA_MULTIARCH void setCenter(const VectorType& _q) { m_center = _q; }
        PONCA_MULTIARCH void rescale(Scalar _s)
        {
            if (_s > 0)
            {
                m_scale  = _s;
                m_iScale = 1. / m_scale;
            }
        }

        PONCA_MULTIARCH ScaleVector scale() const { return m_scale; }
        PONCA_MULTIARCH ScaleVector iScale() const { return m_iScale; }

        // Distance from the origin for a point in local coordinates
        PONCA_MULTIARCH Scalar localDistance(const VectorType& _q) const { return _q.norm(); }

        // Gradient of distance from origin in space for a point in local coordinates
        // This version also expect the distance computed by localDistance, which can help reduce computationnal cost.
        PONCA_MULTIARCH VectorType localDistancedw(const VectorType& _q, Scalar _d) const { return _q * m_iScale / _d; }
        PONCA_MULTIARCH MatrixType localDistanced2w(const VectorType& _q, const VectorType& _g, Scalar _d) const
        {
            Scalar hScale = -m_iScale * m_iScale / (_d * _d * _d);

            MatrixType H = hScale * (_q * _q.transpose());
            H.diagonal() += m_iScale * _g;

            return H;
        }

        PONCA_MULTIARCH ScaleVector localDistanceds(const VectorType& _q, Scalar _d) const { return -m_iScale * _d; }
        PONCA_MULTIARCH ScaleMatrix localDistanced2s(const VectorType& _q, const ScaleVector& _g, Scalar _d) const
        {
            return -2. * m_iScale * _g;
        }

        PONCA_MULTIARCH VectorType localDistancedsdw(const VectorType& _q, Scalar _d) const
        {
            return -m_iScale * localDistancedw(_q, _d);
        }

        PONCA_SPACE_FRAME_FUNCTIONS
        PONCA_SPACE_FRAME_DERIVATIVES
    protected:
        VectorType m_center;
        Scalar m_scale;
        Scalar m_iScale;
    };
} // namespace Ponca
