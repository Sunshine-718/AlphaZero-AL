#ifndef BA59939E_F983_4437_B717_F6728389B8C0
#define BA59939E_F983_4437_B717_F6728389B8C0
#pragma once

namespace AlphaZero {
    struct Config {
        static constexpr int ROWS = 6;
        static constexpr int COLS = 7;
        static constexpr int ACTION_SIZE = 7;
        static constexpr float NOISE_EPSILON = 0.25f;
        static constexpr float NOISE_ALPHA = 1.25f;
    };
}

#endif /* BA59939E_F983_4437_B717_F6728389B8C0 */
