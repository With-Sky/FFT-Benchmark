#include <vector>
#include <array>
#include <complex>
#include <iostream>
#include <future>
#include <ctime>
#include <cstring>
#include "hint_simd.hpp"
#include "stopwatch.hpp"

#define MULTITHREAD 0   // 多线程 0 means no, 1 means yes
#define TABLE_PRELOAD 1 // 是否提前初始化表 0 means no, 1 means yes

namespace hint
{
    using UINT_8 = uint8_t;
    using UINT_16 = uint16_t;
    using UINT_32 = uint32_t;
    using UINT_64 = uint64_t;
    using INT_32 = int32_t;
    using INT_64 = int64_t;
    using ULONG = unsigned long;
    using LONG = long;
    using HintFloat = double;
    using Complex = std::complex<HintFloat>;

    constexpr UINT_64 HINT_CHAR_BIT = 8;
    constexpr UINT_64 HINT_SHORT_BIT = 16;
    constexpr UINT_64 HINT_INT_BIT = 32;
    constexpr UINT_64 HINT_INT8_0XFF = UINT8_MAX;
    constexpr UINT_64 HINT_INT8_0X10 = (UINT8_MAX + 1ull);
    constexpr UINT_64 HINT_INT16_0XFF = UINT16_MAX;
    constexpr UINT_64 HINT_INT16_0X10 = (UINT16_MAX + 1ull);
    constexpr UINT_64 HINT_INT32_0XFF = UINT32_MAX;
    constexpr UINT_64 HINT_INT32_0X01 = 1;
    constexpr UINT_64 HINT_INT32_0X80 = 0X80000000ull;
    constexpr UINT_64 HINT_INT32_0X7F = INT32_MAX;
    constexpr UINT_64 HINT_INT32_0X10 = (UINT32_MAX + 1ull);
    constexpr UINT_64 HINT_INT64_0X80 = INT64_MIN;
    constexpr UINT_64 HINT_INT64_0X7F = INT64_MAX;
    constexpr UINT_64 HINT_INT64_0XFF = UINT64_MAX;

    constexpr double HINT_PI = 3.1415926535897932384626433832795;
    constexpr double HINT_2PI = HINT_PI * 2;
    constexpr double HINT_HSQ_ROOT2 = 0.70710678118654752440084436210485;

    constexpr UINT_64 NTT_MOD1 = 3221225473;
    constexpr UINT_64 NTT_ROOT1 = 5;
    constexpr UINT_64 NTT_MOD2 = 3489660929;
    constexpr UINT_64 NTT_ROOT2 = 3;
    constexpr size_t NTT_MAX_LEN = 1ull << 28;

    /// @brief 生成不大于n的最大的2的幂次的数
    /// @param n
    /// @return 不大于n的最大的2的幂次的数
    template <typename T>
    constexpr T max_2pow(T n)
    {
        T res = 1;
        res <<= (sizeof(T) * 8 - 1);
        while (res > n)
        {
            res /= 2;
        }
        return res;
    }
    /// @brief 生成不小于n的最小的2的幂次的数
    /// @param n
    /// @return 不小于n的最小的2的幂次的数
    template <typename T>
    constexpr T min_2pow(T n)
    {
        T res = 1;
        while (res < n)
        {
            res *= 2;
        }
        return res;
    }
    template <typename T>
    constexpr T hint_log2(T n)
    {
        T res = 0;
        while (n > 1)
        {
            n /= 2;
            res++;
        }
        return res;
    }
#if MULTITHREAD == 1
    const UINT_32 hint_threads = std::thread::hardware_concurrency();
    const UINT_32 log2_threads = std::ceil(hint_log2(hint_threads));
    std::atomic<UINT_32> cur_ths;
#endif

    // 模板数组拷贝
    template <typename T>
    void ary_copy(T *target, const T *source, size_t len)
    {
        if (len == 0 || target == source)
        {
            return;
        }
        if (len >= INT64_MAX)
        {
            throw("Ary too long\n");
        }
        std::memcpy(target, source, len * sizeof(T));
    }
    // 重分配空间
    template <typename T>
    inline T *ary_realloc(T *ptr, size_t len)
    {
        if (len * sizeof(T) < INT64_MAX)
        {
            ptr = static_cast<T *>(realloc(ptr, len * sizeof(T)));
        }
        if (ptr == nullptr)
        {
            throw("realloc error");
        }
        return ptr;
    }
    // 从其他类型数组拷贝到复数组
    template <typename T>
    inline void com_ary_combine_copy(Complex *target, const T &source1, size_t len1, const T &source2, size_t len2)
    {
        size_t min_len = std::min(len1, len2);
        size_t i = 0;
        while (i < min_len)
        {
            target[i] = Complex(source1[i], source2[i]);
            i++;
        }
        while (i < len1)
        {
            target[i].real(source1[i]);
            i++;
        }
        while (i < len2)
        {
            target[i].imag(source2[i]);
            i++;
        }
    }
    // 数组交错重排
    template <UINT_64 N, typename T>
    void ary_interlace(T ary[], size_t len)
    {
        size_t sub_len = len / N;
        T *tmp_ary = new T[len - sub_len];
        for (size_t i = 0; i < len; i += N)
        {
            ary[i / N] = ary[i];
            for (size_t j = 0; j < N - 1; j++)
            {
                tmp_ary[j * sub_len + i / N] = ary[i + j + 1];
            }
        }
        ary_copy(ary + sub_len, tmp_ary, len - sub_len);
        delete[] tmp_ary;
    }
    // FFT与类FFT变换的命名空间
    namespace hint_transform
    {
        class ComplexTableY
        {
        private:
            std::vector<std::vector<Complex>> table1;
            std::vector<std::vector<Complex>> table3;
            INT_32 max_log_size = 2;
            INT_32 cur_log_size = 2;

            static constexpr size_t FAC = 1;

            ComplexTableY(const ComplexTableY &) = delete;
            ComplexTableY &operator=(const ComplexTableY &) = delete;

        public:
            ~ComplexTableY() {}
            // 初始化可以生成平分圆1<<shift份产生的单位根的表
            ComplexTableY(UINT_32 max_shift)
            {
                max_shift = std::max<size_t>(max_shift, 1);
                max_log_size = max_shift;
                table1.resize(max_shift + 1);
                table3.resize(max_shift + 1);
                table1[0] = table1[1] = table3[0] = table3[1] = std::vector<Complex>{1};
                table1[2] = table3[2] = std::vector<Complex>{1};
#if TABLE_PRELOAD == 1
                expand(max_shift);
#endif
            }
            void expand(INT_32 shift)
            {
                shift = std::max<INT_32>(shift, 2);
                if (shift > max_log_size)
                {
                    throw("FFT length too long for lut\n");
                }
                for (INT_32 i = cur_log_size + 1; i <= shift; i++)
                {
                    size_t len = 1ull << i, vec_size = len * FAC / 4;
                    table1[i].resize(vec_size);
                    table3[i].resize(vec_size);
                    table1[i][0] = table3[i][0] = Complex(1, 0);
                    for (size_t pos = 0; pos < vec_size / 2; pos++)
                    {
                        table1[i][pos * 2] = table1[i - 1][pos];
                        table3[i][pos * 2] = table3[i - 1][pos];
                    }
                    for (size_t pos = 1; pos < vec_size / 2; pos += 2)
                    {
                        Complex tmp = unit_root(-HINT_2PI * pos / len);
                        table1[i][pos] = tmp;
                        table1[i][vec_size - pos] = -Complex(tmp.imag(), tmp.real());
                    }
                    table1[i][vec_size / 2] = std::conj(unit_root(8, 1));
                    for (size_t pos = 1; pos < vec_size / 2; pos += 2)
                    {
                        Complex tmp;
                        if (pos * 3 < vec_size)
                        {
                            tmp = table1[i][pos * 3];
                        }
                        else
                        {
                            tmp = table1[i][vec_size * 2 - pos * 3];
                            tmp.real(-tmp.real());
                        }
                        table3[i][pos] = tmp;
                        table3[i][vec_size - pos] = Complex(tmp.imag(), tmp.real());
                    }
                    table3[i][vec_size / 2] = std::conj(unit_root(8, 3));
                }
                cur_log_size = std::max(cur_log_size, shift);
            }
            // 返回单位圆上辐角为theta的点
            static Complex unit_root(double theta)
            {
                return std::polar<double>(1.0, theta);
            }
            // 返回单位圆上平分m份的第n个
            static Complex unit_root(size_t m, size_t n)
            {
                return unit_root((HINT_2PI * n) / m);
            }
            // shift表示圆平分为1<<shift份,3n表示第几个单位根
            Complex get_omega(UINT_32 shift, size_t n) const
            {
                size_t vec_size = (size_t(1) << shift) / 4;
                if (n < vec_size)
                {
                    return table1[shift][n];
                }
                else if (n > vec_size)
                {
                    Complex tmp = table1[shift][vec_size * 2 - n];
                    return Complex(-tmp.real(), tmp.imag());
                }
                else
                {
                    return Complex(0, -1);
                }
            }
            // shift表示圆平分为1<<shift份,3n表示第几个单位根
            Complex get_omega3(UINT_32 shift, size_t n) const
            {
                return table3[shift][n];
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根
            Complex2 get_omegaX2(UINT_32 shift, size_t n) const
            {
                return Complex2(table1[shift].data() + n);
            }
            // shift表示圆平分为1<<shift份,3n表示第几个单位根
            Complex2 get_omega3X2(UINT_32 shift, size_t n) const
            {
                return Complex2(table3[shift].data() + n);
            }
            // shift表示圆平分为1<<shift份,n表示第几个单位根
            const Complex *get_omega_ptr(UINT_32 shift, size_t n) const
            {
                return table1[shift].data() + n;
            }
            // shift表示圆平分为1<<shift份,3n表示第几个单位根
            const Complex *get_omega3_ptr(UINT_32 shift, size_t n) const
            {
                return table3[shift].data() + n;
            }
        };
        constexpr size_t lut_max_rank = 23;

        static ComplexTableY TABLE(lut_max_rank);

        // 二进制逆序
        template <typename T>
        void binary_reverse_swap(T &ary, size_t len)
        {
            size_t i = 0;
            for (size_t j = 1; j < len - 1; j++)
            {
                size_t k = len >> 1;
                i ^= k;
                while (k > i)
                {
                    k >>= 1;
                    i ^= k;
                };
                if (j < i)
                {
                    std::swap(ary[i], ary[j]);
                }
            }
        }
        // 四进制逆序
        template <typename SizeType = UINT_32, typename T>
        void quaternary_reverse_swap(T &ary, size_t len)
        {
            SizeType log_n = hint_log2(len);
            SizeType *rev = new SizeType[len / 4];
            rev[0] = 0;
            for (SizeType i = 1; i < len; i++)
            {
                SizeType index = (rev[i >> 2] >> 2) | ((i & 3) << (log_n - 2)); // 求rev交换数组
                if (i < len / 4)
                {
                    rev[i] = index;
                }
                if (i < index)
                {
                    std::swap(ary[i], ary[index]);
                }
            }
            delete[] rev;
        }
        // 2点fft
        template <typename T>
        inline void fft_2point(T &sum, T &diff)
        {
            T tmp0 = sum;
            T tmp1 = diff;
            sum = tmp0 + tmp1;
            diff = tmp0 - tmp1;
        }
        // 4点fft
        inline void fft_4point(Complex *input, size_t rank = 1)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp1;
            input[rank] = tmp2 + tmp3;
            input[rank * 2] = tmp0 - tmp1;
            input[rank * 3] = tmp2 - tmp3;
        }
        inline void fft_dit_4point_avx(Complex *input)
        {
            static const __m256d neg_mask = _mm256_castsi256_pd(
                _mm256_set_epi64x(INT64_MIN, 0, 0, 0));
            __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
            __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3

            __m256d tmp2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0,c2
            __m256d tmp3 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c1,c3

            tmp0 = _mm256_add_pd(tmp2, tmp3); // c0+c1,c2+c3
            tmp1 = _mm256_sub_pd(tmp2, tmp3); // c0-c1,c2-c3

            tmp2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0+c1,c0-c1;(A,B)
            tmp3 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c2+c3,c2-c3

            tmp3 = _mm256_permute_pd(tmp3, 0b0110);
            tmp3 = _mm256_xor_pd(tmp3, neg_mask); // (C,D)

            tmp0 = _mm256_add_pd(tmp2, tmp3); // A+C,B+D
            tmp1 = _mm256_sub_pd(tmp2, tmp3); // A-C,B-D

            _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp0);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp1);
        }
        inline void fft_dit_8point_avx(Complex *input)
        {
            static const __m256d neg_mask = _mm256_castsi256_pd(_mm256_set_epi64x(INT64_MIN, 0, 0, 0));
            static const __m256d mul1 = _mm256_set_pd(0.70710678118654752440084436210485, 0.70710678118654752440084436210485, 0, 0);
            static const __m256d mul2 = _mm256_set_pd(-0.70710678118654752440084436210485, -0.70710678118654752440084436210485, -1, 1);
            __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
            __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3
            __m256d tmp2 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 4)); // c0,c1
            __m256d tmp3 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 6)); // c2,c3

            __m256d tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0,c2
            __m256d tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c1,c3
            __m256d tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20); // c0,c2
            __m256d tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31); // c1,c3

            tmp0 = _mm256_add_pd(tmp4, tmp5); // c0+c1,c2+c3
            tmp1 = _mm256_sub_pd(tmp4, tmp5); // c0-c1,c2-c3
            tmp2 = _mm256_add_pd(tmp6, tmp7); // c0+c1,c2+c3
            tmp3 = _mm256_sub_pd(tmp6, tmp7); // c0-c1,c2-c3

            tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20); // c0+c1,c0-c1;(A,B)
            tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31); // c2+c3,c2-c3
            tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20); // c0+c1,c0-c1;(A,B)
            tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31); // c2+c3,c2-c3

            tmp5 = _mm256_permute_pd(tmp5, 0b0110);
            tmp5 = _mm256_xor_pd(tmp5, neg_mask); // (C,D)
            tmp7 = _mm256_permute_pd(tmp7, 0b0110);
            tmp7 = _mm256_xor_pd(tmp7, neg_mask); // (C,D)

            tmp0 = _mm256_add_pd(tmp4, tmp5); // A+C,B+D
            tmp1 = _mm256_sub_pd(tmp4, tmp5); // A-C,B-D
            tmp2 = _mm256_add_pd(tmp6, tmp7); // A+C,B+D
            tmp3 = _mm256_sub_pd(tmp6, tmp7); // A-C,B-D

            // 2X4point-done
            tmp6 = _mm256_permute_pd(tmp2, 0b0110);
            tmp6 = _mm256_addsub_pd(tmp6, tmp2);
            tmp6 = _mm256_permute_pd(tmp6, 0b0110);
            tmp6 = _mm256_mul_pd(tmp6, mul1);
            tmp2 = _mm256_blend_pd(tmp2, tmp6, 0b1100);

            tmp7 = _mm256_permute_pd(tmp3, 0b0101);
            tmp3 = _mm256_addsub_pd(tmp3, tmp7);
            tmp3 = _mm256_blend_pd(tmp7, tmp3, 0b1100);
            tmp3 = _mm256_mul_pd(tmp3, mul2);

            tmp4 = _mm256_add_pd(tmp0, tmp2);
            tmp5 = _mm256_add_pd(tmp1, tmp3);
            tmp6 = _mm256_sub_pd(tmp0, tmp2);
            tmp7 = _mm256_sub_pd(tmp1, tmp3);
            _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp4);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp5);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 4), tmp6);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 6), tmp7);
        }
        inline void fft_dif_4point_avx(Complex *input)
        {
            __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
            __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3

            __m256d tmp2 = _mm256_add_pd(tmp0, tmp1); // c0+c2,c1+c3;
            __m256d tmp3 = _mm256_sub_pd(tmp0, tmp1); // c0-c2,c1-c3;
            tmp3 = _mm256_permute_pd(tmp3, 0b0110);   // c0-c2,r(c1-c3);

            static const __m256d neg_mask = _mm256_castsi256_pd(
                _mm256_set_epi64x(INT64_MIN, 0, 0, 0));
            tmp3 = _mm256_xor_pd(tmp3, neg_mask);

            tmp0 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20); // A,C
            tmp1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31); // B,D

            tmp2 = _mm256_add_pd(tmp0, tmp1); // A+B,C+D
            tmp3 = _mm256_sub_pd(tmp0, tmp1); // A-B,C-D

            tmp0 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
            tmp1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

            _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp0);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp1);
        }
        inline void fft_dif_8point_avx(Complex *input)
        {
            static const __m256d neg_mask = _mm256_castsi256_pd(_mm256_set_epi64x(INT64_MIN, 0, 0, 0));
            static const __m256d mul1 = _mm256_set_pd(0.70710678118654752440084436210485, 0.70710678118654752440084436210485, 0, 0);
            static const __m256d mul2 = _mm256_set_pd(-0.70710678118654752440084436210485, -0.70710678118654752440084436210485, -1, 1);
            __m256d tmp0 = _mm256_loadu_pd(reinterpret_cast<double *>(input));     // c0,c1
            __m256d tmp1 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 2)); // c2,c3
            __m256d tmp2 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 4)); // c4,c5
            __m256d tmp3 = _mm256_loadu_pd(reinterpret_cast<double *>(input + 6)); // c6,c7

            __m256d tmp4 = _mm256_add_pd(tmp0, tmp2);
            __m256d tmp5 = _mm256_add_pd(tmp1, tmp3);
            __m256d tmp6 = _mm256_sub_pd(tmp0, tmp2);
            __m256d tmp7 = _mm256_sub_pd(tmp1, tmp3);

            tmp2 = _mm256_permute_pd(tmp6, 0b0110);
            tmp2 = _mm256_addsub_pd(tmp2, tmp6);
            tmp2 = _mm256_permute_pd(tmp2, 0b0110);
            tmp2 = _mm256_mul_pd(tmp2, mul1);
            tmp6 = _mm256_blend_pd(tmp6, tmp2, 0b1100);

            tmp3 = _mm256_permute_pd(tmp7, 0b0101);
            tmp7 = _mm256_addsub_pd(tmp7, tmp3);
            tmp7 = _mm256_blend_pd(tmp3, tmp7, 0b1100);
            tmp7 = _mm256_mul_pd(tmp7, mul2);

            // 2X4point
            tmp0 = _mm256_add_pd(tmp4, tmp5);
            tmp1 = _mm256_sub_pd(tmp4, tmp5);
            tmp1 = _mm256_permute_pd(tmp1, 0b0110);
            tmp1 = _mm256_xor_pd(tmp1, neg_mask);

            tmp2 = _mm256_add_pd(tmp6, tmp7);
            tmp3 = _mm256_sub_pd(tmp6, tmp7);
            tmp3 = _mm256_permute_pd(tmp3, 0b0110);
            tmp3 = _mm256_xor_pd(tmp3, neg_mask);

            tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
            tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
            tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
            tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

            tmp0 = _mm256_add_pd(tmp4, tmp5);
            tmp1 = _mm256_sub_pd(tmp4, tmp5);
            tmp2 = _mm256_add_pd(tmp6, tmp7);
            tmp3 = _mm256_sub_pd(tmp6, tmp7);

            tmp4 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
            tmp5 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
            tmp6 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
            tmp7 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);

            _mm256_storeu_pd(reinterpret_cast<double *>(input), tmp4);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 2), tmp5);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 4), tmp6);
            _mm256_storeu_pd(reinterpret_cast<double *>(input + 6), tmp7);
        }
        inline void fft_dit_4point(Complex *input)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[1];
            Complex tmp2 = input[2];
            Complex tmp3 = input[3];

            fft_2point(tmp0, tmp1);
            fft_2point(tmp2, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp2;
            input[1] = tmp1 + tmp3;
            input[2] = tmp0 - tmp2;
            input[3] = tmp1 - tmp3;
        }
        inline void fft_dit_8point(Complex *input)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[1];
            Complex tmp2 = input[2];
            Complex tmp3 = input[3];
            Complex tmp4 = input[4];
            Complex tmp5 = input[5];
            Complex tmp6 = input[6];
            Complex tmp7 = input[7];
            fft_2point(tmp0, tmp1);
            fft_2point(tmp2, tmp3);
            fft_2point(tmp4, tmp5);
            fft_2point(tmp6, tmp7);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());
            tmp7 = Complex(tmp7.imag(), -tmp7.real());

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            fft_2point(tmp4, tmp6);
            fft_2point(tmp5, tmp7);
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            tmp5 = cos_1_8 * Complex(tmp5.imag() + tmp5.real(), tmp5.imag() - tmp5.real());
            tmp6 = Complex(tmp6.imag(), -tmp6.real());
            tmp7 = -cos_1_8 * Complex(tmp7.real() - tmp7.imag(), tmp7.real() + tmp7.imag());

            input[0] = tmp0 + tmp4;
            input[1] = tmp1 + tmp5;
            input[2] = tmp2 + tmp6;
            input[3] = tmp3 + tmp7;
            input[4] = tmp0 - tmp4;
            input[5] = tmp1 - tmp5;
            input[6] = tmp2 - tmp6;
            input[7] = tmp3 - tmp7;
        }

        inline void fft_dif_4point(Complex *input)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[1];
            Complex tmp2 = input[2];
            Complex tmp3 = input[3];

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp1;
            input[1] = tmp0 - tmp1;
            input[2] = tmp2 + tmp3;
            input[3] = tmp2 - tmp3;
        }
        inline void fft_dif_8point(Complex *input)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[1];
            Complex tmp2 = input[2];
            Complex tmp3 = input[3];
            Complex tmp4 = input[4];
            Complex tmp5 = input[5];
            Complex tmp6 = input[6];
            Complex tmp7 = input[7];

            fft_2point(tmp0, tmp4);
            fft_2point(tmp1, tmp5);
            fft_2point(tmp2, tmp6);
            fft_2point(tmp3, tmp7);
            static constexpr double cos_1_8 = 0.70710678118654752440084436210485;
            tmp5 = cos_1_8 * Complex(tmp5.imag() + tmp5.real(), tmp5.imag() - tmp5.real());
            tmp6 = Complex(tmp6.imag(), -tmp6.real());
            tmp7 = -cos_1_8 * Complex(tmp7.real() - tmp7.imag(), tmp7.real() + tmp7.imag());

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            fft_2point(tmp4, tmp6);
            fft_2point(tmp5, tmp7);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());
            tmp7 = Complex(tmp7.imag(), -tmp7.real());

            input[0] = tmp0 + tmp1;
            input[1] = tmp0 - tmp1;
            input[2] = tmp2 + tmp3;
            input[3] = tmp2 - tmp3;
            input[4] = tmp4 + tmp5;
            input[5] = tmp4 - tmp5;
            input[6] = tmp6 + tmp7;
            input[7] = tmp6 - tmp7;
        }

        // fft基2时间抽取蝶形变换
        inline void fft_radix2_dit_butterfly(Complex omega, Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank] * omega;
            input[0] = tmp0 + tmp1;
            input[rank] = tmp0 - tmp1;
        }
        // fft基2频率抽取蝶形变换
        inline void fft_radix2_dif_butterfly(Complex omega, Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            input[0] = tmp0 + tmp1;
            input[rank] = (tmp0 - tmp1) * omega;
        }
        // fft基2频率抽取蝶形变换
        inline void fft_radix2_dif_butterfly(Complex2 omega, Complex *input, size_t rank)
        {
            Complex2 tmp0(input);
            Complex2 tmp1(input + rank);
            (tmp0 + tmp1).store(input);
            ((tmp0 - tmp1) * omega).store(input + rank);
        }

        // fft分裂基时间抽取蝶形变换
        inline void fft_split_radix_dit_butterfly(Complex omega, Complex omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2] * omega;
            Complex tmp3 = input[rank * 3] * omega_cube;

            fft_2point(tmp2, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp2;
            input[rank] = tmp1 + tmp3;
            input[rank * 2] = tmp0 - tmp2;
            input[rank * 3] = tmp1 - tmp3;
        }
        // fft分裂基频率抽取蝶形变换
        inline void fft_split_radix_dif_butterfly(Complex omega, Complex omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0;
            input[rank] = tmp1;
            input[rank * 2] = (tmp2 + tmp3) * omega;
            input[rank * 3] = (tmp2 - tmp3) * omega_cube;
        }
        // fft分裂基时间抽取蝶形变换
        inline void fft_split_radix_dit_butterfly(Complex2 omega, Complex2 omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex2 tmp0 = input;
            Complex2 tmp1 = input + rank;
            Complex2 tmp2 = Complex2(input + rank * 2) * omega;
            Complex2 tmp3 = Complex2(input + rank * 3) * omega_cube;

            fft_2point(tmp2, tmp3);
            tmp3 = tmp3.mul_neg_i();

            (tmp0 + tmp2).store(input);
            (tmp1 + tmp3).store(input + rank);
            (tmp0 - tmp2).store(input + rank * 2);
            (tmp1 - tmp3).store(input + rank * 3);
        }
        // fft分裂基频率抽取蝶形变换
        inline void fft_split_radix_dif_butterfly(Complex2 omega, Complex2 omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex2 tmp0 = input;
            Complex2 tmp1 = input + rank;
            Complex2 tmp2 = input + rank * 2;
            Complex2 tmp3 = input + rank * 3;

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = tmp3.mul_neg_i();

            tmp0.store(input);
            tmp1.store(input + rank);
            ((tmp2 + tmp3) * omega).store(input + rank * 2);
            ((tmp2 - tmp3) * omega_cube).store(input + rank * 3);
        }
        // fft分裂基时间抽取蝶形变换
        inline void fft_split_radix_dit_butterfly(const Complex *omega, const Complex *omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex2 tmp0 = input;
            Complex2 tmp4 = input + 2;
            Complex2 tmp1 = input + rank;
            Complex2 tmp5 = input + rank + 2;
            Complex2 tmp2 = Complex2(input + rank * 2) * Complex2(omega);
            Complex2 tmp6 = Complex2(input + rank * 2 + 2) * Complex2(omega + 2);
            Complex2 tmp3 = Complex2(input + rank * 3) * Complex2(omega_cube);
            Complex2 tmp7 = Complex2(input + rank * 3 + 2) * Complex2(omega_cube + 2);

            fft_2point(tmp2, tmp3);
            fft_2point(tmp6, tmp7);
            tmp3 = tmp3.mul_neg_i();
            tmp7 = tmp7.mul_neg_i();

            (tmp0 + tmp2).store(input);
            (tmp4 + tmp6).store(input + 2);
            (tmp1 + tmp3).store(input + rank);
            (tmp5 + tmp7).store(input + rank + 2);
            (tmp0 - tmp2).store(input + rank * 2);
            (tmp4 - tmp6).store(input + rank * 2 + 2);
            (tmp1 - tmp3).store(input + rank * 3);
            (tmp5 - tmp7).store(input + rank * 3 + 2);
        }
        // fft分裂基频率抽取蝶形变换
        inline void fft_split_radix_dif_butterfly(const Complex *omega, const Complex *omega_cube,
                                                  Complex *input, size_t rank)
        {
            Complex2 tmp0 = input;
            Complex2 tmp4 = input + 2;
            Complex2 tmp1 = input + rank;
            Complex2 tmp5 = input + rank + 2;
            Complex2 tmp2 = input + rank * 2;
            Complex2 tmp6 = input + rank * 2 + 2;
            Complex2 tmp3 = input + rank * 3;
            Complex2 tmp7 = input + rank * 3 + 2;

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            fft_2point(tmp4, tmp6);
            fft_2point(tmp5, tmp7);
            tmp3 = tmp3.mul_neg_i();
            tmp7 = tmp7.mul_neg_i();

            tmp0.store(input);
            tmp4.store(input + 2);
            tmp1.store(input + rank);
            tmp5.store(input + rank + 2);
            ((tmp2 + tmp3) * Complex2(omega)).store(input + rank * 2);
            ((tmp6 + tmp7) * Complex2(omega + 2)).store(input + rank * 2 + 2);
            ((tmp2 - tmp3) * Complex2(omega_cube)).store(input + rank * 3);
            ((tmp6 - tmp7) * Complex2(omega_cube + 2)).store(input + rank * 3 + 2);
        }
        // fft基4时间抽取蝶形变换
        inline void fft_radix4_dit_butterfly(Complex omega, Complex omega_sqr, Complex omega_cube,
                                             Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank] * omega;
            Complex tmp2 = input[rank * 2] * omega_sqr;
            Complex tmp3 = input[rank * 3] * omega_cube;

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp1;
            input[rank] = tmp2 + tmp3;
            input[rank * 2] = tmp0 - tmp1;
            input[rank * 3] = tmp2 - tmp3;
        }
        // fft基4频率抽取蝶形变换
        inline void fft_radix4_dif_butterfly(Complex omega, Complex omega_sqr, Complex omega_cube,
                                             Complex *input, size_t rank)
        {
            Complex tmp0 = input[0];
            Complex tmp1 = input[rank];
            Complex tmp2 = input[rank * 2];
            Complex tmp3 = input[rank * 3];

            fft_2point(tmp0, tmp2);
            fft_2point(tmp1, tmp3);
            tmp3 = Complex(tmp3.imag(), -tmp3.real());

            input[0] = tmp0 + tmp1;
            input[rank] = (tmp2 + tmp3) * omega;
            input[rank * 2] = (tmp0 - tmp1) * omega_sqr;
            input[rank * 3] = (tmp2 - tmp3) * omega_cube;
        }
        // 求共轭复数及归一化，逆变换用
        inline void fft_conj(Complex *input, size_t fft_len, double div = 1)
        {
            for (size_t i = 0; i < fft_len; i++)
            {
                input[i] = std::conj(input[i]) / div;
            }
        }
        // 归一化,逆变换用
        inline void fft_normalize(Complex *input, size_t fft_len)
        {
            double len = static_cast<double>(fft_len);
            for (size_t i = 0; i < fft_len; i++)
            {
                input[i] /= len;
            }
        }
        // 模板化时间抽取分裂基fft
        template <size_t LEN>
        void fft_split_radix_dit_template(Complex *input)
        {
            constexpr size_t log_len = hint_log2(LEN);
            constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
            fft_split_radix_dit_template<half_len>(input);
            fft_split_radix_dit_template<quarter_len>(input + half_len);
            fft_split_radix_dit_template<quarter_len>(input + half_len + quarter_len);
            for (size_t i = 0; i < quarter_len; i += 4)
            {
                auto omega = TABLE.get_omega_ptr(log_len, i);
                auto omega_cube = TABLE.get_omega3_ptr(log_len, i);
                fft_split_radix_dit_butterfly(omega, omega_cube, input + i, quarter_len);
            }
        }
        template <>
        void fft_split_radix_dit_template<0>(Complex *input) {}
        template <>
        void fft_split_radix_dit_template<1>(Complex *input) {}
        template <>
        void fft_split_radix_dit_template<2>(Complex *input)
        {
            fft_2point(input[0], input[1]);
        }
        template <>
        void fft_split_radix_dit_template<4>(Complex *input)
        {
            fft_dit_4point_avx(input);
        }
        template <>
        void fft_split_radix_dit_template<8>(Complex *input)
        {
            fft_dit_8point_avx(input);
        }

        // 模板化频率抽取分裂基fft
        template <size_t LEN>
        void fft_split_radix_dif_template(Complex *input)
        {
            constexpr size_t log_len = hint_log2(LEN);
            constexpr size_t half_len = LEN / 2, quarter_len = LEN / 4;
            for (size_t i = 0; i < quarter_len; i += 4)
            {
                auto omega = TABLE.get_omega_ptr(log_len, i);
                auto omega_cube = TABLE.get_omega3_ptr(log_len, i);
                fft_split_radix_dif_butterfly(omega, omega_cube, input + i, quarter_len);
            }
            fft_split_radix_dif_template<half_len>(input);
            fft_split_radix_dif_template<quarter_len>(input + half_len);
            fft_split_radix_dif_template<quarter_len>(input + half_len + quarter_len);
        }
        template <>
        void fft_split_radix_dif_template<0>(Complex *input) {}
        template <>
        void fft_split_radix_dif_template<1>(Complex *input) {}
        template <>
        void fft_split_radix_dif_template<2>(Complex *input)
        {
            fft_2point(input[0], input[1]);
        }
        template <>
        void fft_split_radix_dif_template<4>(Complex *input)
        {
            fft_dif_4point_avx(input);
        }
        template <>
        void fft_split_radix_dif_template<8>(Complex *input)
        {
            fft_dif_8point_avx(input);
        }

        template <size_t LEN = 1>
        void fft_dit_template(Complex *input, size_t fft_len)
        {
            if (fft_len > LEN)
            {
                fft_dit_template<LEN * 2>(input, fft_len);
                return;
            }
            TABLE.expand(hint_log2(LEN));
            fft_split_radix_dit_template<LEN>(input);
        }
        template <>
        void fft_dit_template<1 << 24>(Complex *input, size_t fft_len) {}

        template <size_t LEN = 1>
        void fft_dif_template(Complex *input, size_t fft_len)
        {
            if (fft_len > LEN)
            {
                fft_dif_template<LEN * 2>(input, fft_len);
                return;
            }
            TABLE.expand(hint_log2(LEN));
            fft_split_radix_dif_template<LEN>(input);
        }
        template <>
        void fft_dif_template<1 << 24>(Complex *input, size_t fft_len) {}

        /// @brief 时间抽取基2fft
        /// @param input 复数组
        /// @param fft_len 数组长度
        /// @param bit_rev 是否逆序
        inline void fft_dit(Complex *input, size_t fft_len, bool bit_rev = true)
        {
            fft_len = max_2pow(fft_len);
            if (bit_rev)
            {
                binary_reverse_swap(input, fft_len);
            }
            fft_dit_template<1>(input, fft_len);
        }

        /// @brief 频率抽取基2fft
        /// @param input 复数组
        /// @param fft_len 数组长度
        /// @param bit_rev 是否逆序
        inline void fft_dif(Complex *input, size_t fft_len, bool bit_rev = true)
        {
            fft_len = max_2pow(fft_len);
            fft_dif_template<1>(input, fft_len);
            if (bit_rev)
            {
                binary_reverse_swap(input, fft_len);
            }
        }
        /// @brief 快速傅里叶变换
        /// @param input 复数组
        /// @param fft_len 变换长度
        /// @param r4_bit_rev 基4是否进行比特逆序,与逆变换同时设为false可以提高性能
        inline void fft(Complex *input, size_t fft_len, const bool bit_rev = true)
        {
            size_t log_len = hint_log2(fft_len);
            fft_len = 1ull << log_len;
            if (fft_len <= 1)
            {
                return;
            }
            fft_dif(input, fft_len, bit_rev);
        }
        /// @brief 快速傅里叶逆变换
        /// @param input 复数组
        /// @param fft_len 变换长度
        /// @param r4_bit_rev 基4是否进行比特逆序,与逆变换同时设为false可以提高性能
        inline void ifft(Complex *input, size_t fft_len, const bool bit_rev = true)
        {
            size_t log_len = hint_log2(fft_len);
            fft_len = 1ull << log_len;
            if (fft_len <= 1)
            {
                return;
            }
            fft_len = max_2pow(fft_len);
            fft_conj(input, fft_len);
            fft_dit(input, fft_len, bit_rev);
            fft_conj(input, fft_len, fft_len);
        }
#if MULTITHREAD == 1
        void fft_dit_2ths(Complex *input, size_t fft_len)
        {
            const size_t half_len = fft_len / 2;
            const INT_32 log_len = hint_log2(fft_len);
            TABLE.expand(log_len);
            auto th = std::async(fft_dit, input, half_len, false);
            fft_dit(input + half_len, half_len, false);
            th.wait();
            auto proc = [&](size_t start, size_t end)
            {
                for (size_t i = start; i < end; i++)
                {
                    Complex omega = TABLE.get_omega(log_len, i);
                    fft_radix2_dit_butterfly(omega, input + i, half_len);
                }
            };
            th = std::async(proc, 0, half_len / 2);
            proc(half_len / 2, half_len);
            th.wait();
        }
        void fft_dif_2ths(Complex *input, size_t fft_len)
        {
            const size_t half_len = fft_len / 2;
            const INT_32 log_len = hint_log2(fft_len);
            TABLE.expand(log_len);
            auto proc = [&](size_t start, size_t end)
            {
                for (size_t i = start; i < end; i++)
                {
                    Complex omega = TABLE.get_omega(log_len, i);
                    fft_radix2_dif_butterfly(omega, input + i, half_len);
                }
            };
            auto th = std::async(proc, 0, half_len / 2);
            proc(half_len / 2, half_len);
            th.wait();
            th = std::async(fft_dif, input, half_len, false);
            fft_dif(input + half_len, half_len, false);
            th.wait();
        }
        void fft_dit_4ths(Complex *input, size_t fft_len)
        {
            const size_t half_len = fft_len / 2;
            const INT_32 log_len = hint_log2(fft_len);
            TABLE.expand(log_len);
            auto th1 = std::async(fft_dit_2ths, input, half_len);
            fft_dit_2ths(input + half_len, half_len);
            th1.wait();

            auto proc = [&](size_t start, size_t end)
            {
                for (size_t i = start; i < end; i++)
                {
                    Complex omega = TABLE.get_omega(log_len, i);
                    fft_radix2_dit_butterfly(omega, input + i, half_len);
                }
            };
            const size_t sub_len = fft_len / 8;
            th1 = std::async(proc, 0, sub_len);
            auto th2 = std::async(proc, sub_len, sub_len * 2);
            auto th3 = std::async(proc, sub_len * 2, sub_len * 3);
            proc(sub_len * 3, sub_len * 4);
            th1.wait();
            th2.wait();
            th3.wait();
        }
        void fft_dif_4ths(Complex *input, size_t fft_len)
        {
            const size_t half_len = fft_len / 2;
            const INT_32 log_len = hint_log2(fft_len);
            TABLE.expand(log_len);
            auto proc = [&](size_t start, size_t end)
            {
                for (size_t i = start; i < end; i++)
                {
                    Complex omega = TABLE.get_omega(log_len, i);
                    fft_radix2_dif_butterfly(omega, input + i, half_len);
                }
            };
            const size_t sub_len = fft_len / 8;
            auto th1 = std::async(proc, 0, sub_len);
            auto th2 = std::async(proc, sub_len, sub_len * 2);
            auto th3 = std::async(proc, sub_len * 2, sub_len * 3);
            proc(sub_len * 3, sub_len * 4);
            th1.wait();
            th2.wait();
            th3.wait();

            th1 = std::async(fft_dif_2ths, input, half_len);
            fft_dif_2ths(input + half_len, half_len);
            th1.wait();
        }
#endif
    }
}
using namespace std;
using namespace hint;
using namespace hint_transform;

template <typename T>
vector<T> poly_multiply(const vector<T> &in1, const vector<T> &in2)
{
    size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
    vector<T> result(out_len);
    size_t fft_len = min_2pow(out_len);
    // Complex2 *avx_ary = new Complex2[fft_len / 2];
    Complex *fft_ary = (Complex *)_mm_malloc(fft_len * sizeof(Complex), sizeof(Complex2));
    com_ary_combine_copy(fft_ary, in1, len1, in2, len2);
    // fft_radix2_dif_lut(fft_ary, fft_len, false); // 经典FFT
#if MULTITHREAD == 1
    fft_dif_4ths(fft_ary, fft_len); // 4线程FFT
#else
    fft_dif(fft_ary, fft_len, false); // 优化FFT
#endif
    double inv = -0.5 / fft_len;
    Complex2 invx4(inv);
    for (size_t i = 0; i < fft_len; i++)
    {
        Complex tmp = fft_ary[i];
        fft_ary[i] = std::conj(tmp * tmp) * inv;
    }
    // fft_radix2_dit_lut(fft_ary, fft_len, false); // 经典FFT
#if MULTITHREAD == 1
    fft_dit_4ths(fft_ary, fft_len); // 4线程FFT
#else
    fft_dit(fft_ary, fft_len, false); // 优化FFT
#endif
    for (size_t i = 0; i < out_len; i++)
    {
        result[i] = static_cast<T>(fft_ary[i].imag() + 0.5);
    }
    _mm_free(fft_ary);
    return result;
}
inline void stress(Complex ary[], size_t len)
{
    for (size_t i = 0; i < 1000000000; i++)
    {
        fft_dif(ary, len, false);
        fft_dit(ary, len, false);
    }
}
inline void mthstress(Complex ary[], size_t len, int ths)
{
    vector<future<void>> v(ths);
    for (auto &&i : v)
    {
        i = std::async(stress, ary, len);
    }
    for (auto &&i : v)
    {
        i.wait();
    }
}
template <typename T>
void result_test(const vector<T> &res, T ele)
{
    size_t len = res.size();
    for (size_t i = 0; i < len / 2; i++)
    {
        uint64_t x = (i + 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            cout << "fail:" << i << "\t" << (i + 1) * ele * ele << "\t" << y << "\n";
            return;
        }
    }
    for (size_t i = len / 2; i < len; i++)
    {
        uint64_t x = (len - i - 1) * ele * ele;
        uint64_t y = res[i];
        if (x != y)
        {
            cout << "fail:" << i << "\t" << x << "\t" << y << "\n";
            return;
        }
    }
    cout << "success\n";
}

int main()
{
    StopWatch w(1000);
    int n = 18;
    cin >> n;
    size_t len = 1 << n; // 变换长度
    cout << "fft len:" << len << "\n";
    uint64_t ele = 9;
    vector<uint32_t> in1(len / 2, ele);
    vector<uint32_t> in2(len / 2, ele); // 计算两个长度为len/2，每个元素为ele的卷积
    w.start();
    vector<uint32_t> res = poly_multiply(in1, in2);
    w.stop();
    result_test<uint32_t>(res, ele); // 结果校验
    cout << w.duration() << "ms\n";
}