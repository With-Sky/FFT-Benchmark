// 本文件用于测试基2迭代FFT
#include <climits>
#include <complex>
#include <iostream>
#include <vector>
#include "stopwatch.hpp"

using UINT_8 = uint8_t;
using UINT_16 = uint16_t;
using UINT_32 = uint32_t;
using UINT_64 = uint64_t;
using INT_32 = int32_t;
using INT_64 = int64_t;
using ULONG = unsigned long;
using LONG = long;
using Complex = std::complex<double>;
constexpr double HINT_PI = 3.1415926535897932384626433832795;
constexpr double HINT_2PI = HINT_PI * 2;

#define TABLE_PRELOAD 1 // 查找表提前初始化

// 从其他类型数组拷贝到复数组
template <typename T>
inline void com_ary_combine_copy(Complex *target, const T &source1, size_t len1,
								 const T &source2, size_t len2)
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
template <typename T>
constexpr size_t hint_log2(T n)
{
	T res = 0;
	while (n > 1)
	{
		n /= 2;
		res++;
	}
	return res;
}
class ComplexTableX
{
private:
	std::vector<std::vector<Complex>> table;
	INT_32 max_log_size = 2;
	INT_32 cur_log_size = 2;

	static constexpr size_t FAC = 3;

	ComplexTableX(const ComplexTableX &) = delete;
	ComplexTableX &operator=(const ComplexTableX &) = delete;

public:
	~ComplexTableX() {}
	// 初始化可以生成平分圆1<<shift份产生的单位根的表
	ComplexTableX(UINT_32 max_shift)
	{
		max_shift = std::max<size_t>(max_shift, cur_log_size);
		max_log_size = max_shift;
		table.resize(max_shift + 1);
		table[0] = table[1] = std::vector<Complex>{1};
		table[2] =
			std::vector<Complex>{Complex(1, 0), Complex(0, -1), Complex(-1, 0)};
#if TABLE_PRELOAD == 1
		expand(max_shift);
#endif
	}
	void expand(INT_32 shift)
	{
		if (shift > max_log_size)
		{
			throw("FFT length too long for lut\n");
		}
		for (INT_32 i = cur_log_size + 1; i <= shift; i++)
		{
			size_t len = 1ull << i, vec_size = len * FAC / 4;
			table[i].resize(vec_size);
			for (size_t pos = 0; pos < vec_size / 2; pos++) // 有一半的元素可以复用
			{
				table[i][pos * 2] = table[i - 1][pos];
			}
			for (size_t pos = 1; pos < len / 4; pos += 2)
			{
				Complex tmp = std::conj(unit_root(len, pos));
				table[i][pos] = tmp;
				table[i][pos + len / 4] = Complex(tmp.imag(), -tmp.real());
				table[i][pos + len / 2] = -tmp;
			}
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
	// shift表示圆平分为1<<shift份,n表示第几个单位根
	Complex get_complex(UINT_32 shift, size_t n) const
	{
		return std::conj(table[shift][n]);
	}
	// shift表示圆平分为1<<shift份,n表示第几个单位根的共轭
	Complex get_complex_conj(UINT_32 shift, size_t n) const
	{
		return table[shift][n];
	}
};
ComplexTableX TABLE(23);
// fft基2时间抽取蝶形变换
inline void fft_radix2_dit_butterfly(Complex omega, Complex *input,
									 size_t rank)
{
	Complex tmp0 = input[0];
	Complex tmp1 = input[rank] * omega;
	input[0] = tmp0 + tmp1;
	input[rank] = tmp0 - tmp1;
}
// fft基2频率抽取蝶形变换
inline void fft_radix2_dif_butterfly(Complex omega, Complex *input,
									 size_t rank)
{
	Complex tmp0 = input[0];
	Complex tmp1 = input[rank];
	input[0] = tmp0 + tmp1;
	input[rank] = (tmp0 - tmp1) * omega;
}
// 基2迭代fft
void fft_radix2_dit(Complex *input, size_t fft_len, const bool bit_rev = true)
{
	if (bit_rev)
	{
		binary_reverse_swap(input, fft_len); // 先进行二进制逆序交换
	}
	INT_32 shift = 1;
	for (size_t rank = 1; rank < fft_len; rank *= 2)
	{
		size_t gap = rank * 2;								  // rank代表上一级fft的长度,gap为本级fft长度
		for (size_t begin = 0; begin < fft_len; begin += gap) // begin每次跳跃的长度为gap
		{
			for (size_t pos = begin; pos < begin + rank; pos++)
			{
				Complex omega = TABLE.get_complex_conj(shift, pos - begin);
				fft_radix2_dit_butterfly(omega, input + pos, rank);
			}
		}
		shift++;
	}
}

// 基2迭代频率抽取fft
void fft_radix2_dif(Complex *input, size_t fft_len, bool bit_rev = true)
{
	INT_32 shift = hint_log2(fft_len);
	for (size_t rank = fft_len / 2; rank > 0; rank /= 2)
	{
		size_t gap = rank * 2;								  // rank代表上一级fft的长度,gap为本级fft长度
		for (size_t begin = 0; begin < fft_len; begin += gap) // begin每次跳跃的长度为gap
		{
			for (size_t pos = begin; pos < begin + rank; pos++)
			{
				Complex omega = TABLE.get_complex_conj(shift, pos - begin);
				fft_radix2_dif_butterfly(omega, input + pos, rank);
			}
		}
		shift--;
	}
	if (bit_rev)
	{
		binary_reverse_swap(input, fft_len); // 后进行二进制逆序交换
	}
}

// 正变换
void fft(Complex *input, size_t fft_len)
{
	fft_radix2_dif(input, fft_len, true); // 要进行二进制位逆序才是真正的DFT，这里用dit也可
}

// 逆变换
void ifft(Complex *input, size_t fft_len)
{
	for (size_t i = 0; i < fft_len; i++)
	{
		input[i] = std::conj(input[i]); // 先对每个元素进行共轭
	}
	fft(input, fft_len); // 正变换
	double inv = 1.0 / fft_len;
	for (size_t i = 0; i < fft_len; i++)
	{
		input[i] = std::conj(input[i]) * inv; // 对每个元素进行共轭和归一化
	}
}

using namespace std;
// 多项式乘法
template <typename T>
vector<T> poly_multiply(const vector<T> &in1, const vector<T> &in2)
{
	size_t len1 = in1.size(), len2 = in2.size(), out_len = len1 + len2;
	vector<T> result(out_len);
	size_t fft_len = min_2pow(out_len);
	Complex *fft_ary = new Complex[fft_len];
	com_ary_combine_copy(fft_ary, in1, len1, in2, len2);
	fft_radix2_dif(fft_ary, fft_len, false);
	for (size_t i = 0; i < fft_len; i++)
	{
		Complex tmp = fft_ary[i];
		tmp *= tmp;
		fft_ary[i] = std::conj(tmp);
	}
	fft_radix2_dit(fft_ary, fft_len, false);
	double inv = -0.5 / fft_len;
	for (size_t i = 0; i < out_len; i++)
	{
		result[i] = static_cast<T>(fft_ary[i].imag() * inv + 0.5);
	}
	delete[] fft_ary;
	return result;
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