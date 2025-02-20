#include "Utils.h"


double get_time_s() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec + ts.tv_nsec / 1e9;
    } else {
        return 0;
    }
}

double get_time_ms() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec * 1000 + ts.tv_nsec / 1000000.0;
    } else {
        return 0;
    }
}





