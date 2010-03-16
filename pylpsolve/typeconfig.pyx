# PyLPSolve is an object oriented wrapper for the open source LP
# solver lp_solve. Copyright (C) 2010 Hoyt Koepke.
#
# This library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation; either version 2.1 of
# the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA


from numpy import float64, int32, int64, uint32, uint64

if sizeof(double) == 8:
    npfloat = float64
else:
    raise ImportError("Numpy type mismatch on double.")

if sizeof(int) == 4:
    npint = int32
elif sizeof(int) == 8:
    npint = int64
else:
    raise ImportError("Numpy type mismatch on int.")

if sizeof(size_t) == 4:
    npuint = uint32
elif sizeof(size_t) == 8:
    npuint = uint64
else:
    raise ImportError("Numpy type mismatch on size_t.")
