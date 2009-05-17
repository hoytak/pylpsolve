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
