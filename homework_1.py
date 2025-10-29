# ===== ЗАДАЧА 1 =====
def count_vowels(s):
    vowels = 'aeiou'
    return sum(1 for char in s.lower() if char in vowels)

def test_count_vowels():
    assert count_vowels("hello") == 2
    assert count_vowels("world") == 1
    assert count_vowels("AEIOU") == 5
    assert count_vowels("Python") == 1
    assert count_vowels("") == 0
    assert count_vowels("bcdfg") == 0
    print("Тесты задачи 1 пройдены")

# ===== ЗАДАЧА 2 =====
def is_unique_chars(s):
    return len(s) == len(set(s))

def test_is_unique_chars():
    assert is_unique_chars("abcde") == True
    assert is_unique_chars("hello") == False
    assert is_unique_chars("") == True
    assert is_unique_chars("a") == True
    assert is_unique_chars("abacaba") == False
    print("Тесты задачи 2 пройдены")

# ===== ЗАДАЧА 3 =====
def count_bits(n):
    return bin(n).count('1')

def test_count_bits():
    assert count_bits(0) == 0
    assert count_bits(1) == 1
    assert count_bits(2) == 1
    assert count_bits(7) == 3
    assert count_bits(1023) == 10
    print("Тесты задачи 3 пройдены")

# ===== ЗАДАЧА 4 =====
def magic(n):
    if n < 10:
        return 0
    count = 0
    while n >= 10:
        product = 1
        for digit in str(n):
            product *= int(digit)
        n = product
        count += 1
    return count

def test_magic():
    assert magic(39) == 3
    assert magic(4) == 0
    assert magic(999) == 4
    assert magic(10) == 1
    assert magic(123) == 1
    print("Тесты задачи 4 пройдены")

# ===== ЗАДАЧА 5 =====
def mse(pred, true):
    if len(pred) != len(true):
        raise ValueError("Векторы должны быть одинаковой длины")
    return sum((p - t) ** 2 for p, t in zip(pred, true)) / len(pred)

def test_mse():
    assert mse([1, 2, 3], [1, 2, 3]) == 0
    assert mse([1, 2, 3], [2, 3, 4]) == 1
    assert mse([0, 0, 0], [1, 1, 1]) == 1
    assert mse([1.5, 2.5], [1, 2]) == 0.125
    print("Тесты задачи 5 пройдены")

# ===== ЗАДАЧА 6 =====
def prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    result = []
    for factor in sorted(factors.keys()):
        power = factors[factor]
        if power == 1:
            result.append(f"({factor})")
        else:
            result.append(f"({factor}**{power})")
    return ''.join(result)

def test_prime_factors():
    assert prime_factors(86240) == "(2**5)(5)(7**2)(11)"
    assert prime_factors(2) == "(2)"
    assert prime_factors(10) == "(2)(5)"
    assert prime_factors(100) == "(2**2)(5**2)"
    assert prime_factors(17) == "(17)"
    print("Тесты задачи 6 пройдены")

# ===== ЗАДАЧА 7 =====
def pyramid(number):
    total = 0
    k = 1
    while total < number:
        total += k ** 2
        if total == number:
            return k
        k += 1
    return "It is impossible"

def test_pyramid():
    assert pyramid(1) == 1
    assert pyramid(5) == 2
    assert pyramid(14) == 3
    assert pyramid(30) == 4
    assert pyramid(55) == 5
    assert pyramid(100) == "It is impossible"
    print("Тесты задачи 7 пройдены")

# ===== ЗАДАЧА 8 =====
def is_balanced(n):
    digits = str(n)
    length = len(digits)
    
    if length <= 2:
        return True
        
    mid = length // 2
    if length % 2 == 0:
        left_sum = sum(int(d) for d in digits[:mid-1])
        right_sum = sum(int(d) for d in digits[mid+1:])
    else:
        left_sum = sum(int(d) for d in digits[:mid])
        right_sum = sum(int(d) for d in digits[mid+1:])
    
    return left_sum == right_sum

def test_is_balanced():
    assert is_balanced(12345) == False  # 1+2 ≠ 4+5
    assert is_balanced(123321) == True  # 1+2 = 2+1 (средние 33 не учитываются)
    assert is_balanced(121) == True     # 1 = 1 (средняя 2 не учитывается)
    assert is_balanced(1221) == True    # 1 = 1 (средние 22 не учитываются)
    assert is_balanced(1111) == True
    assert is_balanced(1) == True
    assert is_balanced(12) == True
    print("Тесты задачи 8 пройдены")

# Запуск всех тестов
if __name__ == "__main__":
    test_count_vowels()
    test_is_unique_chars()
    test_count_bits()
    test_magic()
    test_mse()
    test_prime_factors()
    test_pyramid()
    test_is_balanced()
    print("Все тесты пройдены!")