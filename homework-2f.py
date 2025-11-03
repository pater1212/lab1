import numpy as np
import matplotlib.pyplot as plt

# ===== ЗАДАЧА 1 =====
def sum_prod(X, V):
    '''
    X - список матриц (n, n)
    V - список векторов (n, 1)
    Гарантируется, что len(X) == len(V)
    '''
    total = np.zeros_like(X[0] @ V[0])
    for matrix, vector in zip(X, V):
        total += matrix @ vector
    return total

def test_sum_prod():
    # Тест 1
    X1 = [np.array([[1, 2], [3, 4]])]
    V1 = [np.array([[1], [2]])]
    result1 = sum_prod(X1, V1)
    expected1 = np.array([[5], [11]])
    assert np.array_equal(result1, expected1)
    
    # Тест 2
    X2 = [np.eye(2), np.eye(2)]
    V2 = [np.array([[1], [2]]), np.array([[3], [4]])]
    result2 = sum_prod(X2, V2)
    expected2 = np.array([[4], [6]])
    assert np.array_equal(result2, expected2)
    
    print("Тесты задачи 1 пройдены")

# ===== ЗАДАЧА 2 =====
def binarize(M, threshold=0.5):
    return (M > threshold).astype(int)

def test_binarize():
    M = np.array([[0.1, 0.6, 0.3],
                  [0.7, 0.2, 0.9],
                  [0.4, 0.5, 0.8]])
    
    result = binarize(M, 0.5)
    expected = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 0, 1]])
    assert np.array_equal(result, expected)
    
    # Тест с другим порогом
    result2 = binarize(M, 0.3)
    expected2 = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [1, 1, 1]])
    assert np.array_equal(result2, expected2)
    
    print("Тесты задачи 2 пройдены")

# ===== ЗАДАЧА 3 =====
def unique_rows(mat):
    return [np.unique(row) for row in mat]

def unique_columns(mat):
    return [np.unique(col) for col in mat.T]

def test_unique():
    mat = np.array([[1, 2, 1],
                    [3, 3, 3],
                    [1, 2, 3]])
    
    rows_result = unique_rows(mat)
    rows_expected = [np.array([1, 2]), np.array([3]), np.array([1, 2, 3])]
    for res, exp in zip(rows_result, rows_expected):
        assert np.array_equal(res, exp)
    
    cols_result = unique_columns(mat)
    cols_expected = [np.array([1, 3]), np.array([2, 3]), np.array([1, 3])]
    for res, exp in zip(cols_result, cols_expected):
        assert np.array_equal(res, exp)
    
    print("Тесты задачи 3 пройдены")

# ===== ЗАДАЧА 4 =====
def analyze_matrix(m, n):
    # Генерация матрицы
    matrix = np.random.normal(0, 1, (m, n))
    
    # Статистики по строкам
    row_means = np.mean(matrix, axis=1)
    row_vars = np.var(matrix, axis=1)
    
    # Статистики по столбцам
    col_means = np.mean(matrix, axis=0)
    col_vars = np.var(matrix, axis=0)
    
    print(f"Матрица {m}x{n}:")
    print(f"Среднее по строкам: {row_means}")
    print(f"Дисперсия по строкам: {row_vars}")
    print(f"Среднее по столбцам: {col_means}")
    print(f"Дисперсия по столбцам: {col_vars}")
    
    # Построение гистограмм
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Гистограммы для строк
    for i in range(min(3, m)):  # Покажем первые 3 строки
        axes[0, 0].hist(matrix[i], alpha=0.7, label=f'Строка {i+1}')
    axes[0, 0].set_title('Гистограммы строк')
    axes[0, 0].legend()
    
    # Гистограммы для столбцов
    for i in range(min(3, n)):  # Покажем первые 3 столбца
        axes[0, 1].hist(matrix[:, i], alpha=0.7, label=f'Столбец {i+1}')
    axes[0, 1].set_title('Гистограммы столбцов')
    axes[0, 1].legend()
    
    # Распределение средних по строкам
    axes[1, 0].hist(row_means, alpha=0.7, color='red')
    axes[1, 0].set_title('Распределение средних по строкам')
    
    # Распределение средних по столбцам
    axes[1, 1].hist(col_means, alpha=0.7, color='green')
    axes[1, 1].set_title('Распределение средних по столбцам')
    
    plt.tight_layout()
    plt.show()
    
    return matrix, row_means, row_vars, col_means, col_vars

# ===== ЗАДАЧА 5 =====
def chess(m, n, a, b):
    board = np.full((m, n), a)
    board[1::2, ::2] = b  # Четные строки, нечетные столбцы
    board[::2, 1::2] = b  # Нечетные строки, четные столбцы
    return board

def test_chess():
    result1 = chess(2, 2, 0, 1)
    expected1 = np.array([[0, 1],
                          [1, 0]])
    assert np.array_equal(result1, expected1)
    
    result2 = chess(3, 3, 'A', 'B')
    expected2 = np.array([['A', 'B', 'A'],
                          ['B', 'A', 'B'],
                          ['A', 'B', 'A']])
    assert np.array_equal(result2, expected2)
    
    print("Тесты задачи 5 пройдены")

# ===== ЗАДАЧА 6 =====
def draw_rectangle(a, b, m, n, rectangle_color, background_color):
    image = np.full((m, n, 3), background_color)
    
    # Центр изображения
    center_x, center_y = n // 2, m // 2
    
    # Координаты прямоугольника
    x_start = center_x - a // 2
    x_end = center_x + a // 2
    y_start = center_y - b // 2
    y_end = center_y + b // 2
    
    # Отрисовка прямоугольника
    image[y_start:y_end, x_start:x_end] = rectangle_color
    
    return image

def draw_ellipse(a, b, m, n, ellipse_color, background_color):
    image = np.full((m, n, 3), background_color)
    
    # Центр изображения
    x0, y0 = n // 2, m // 2
    
    # Создаем сетку координат
    y, x = np.ogrid[:m, :n]
    
    # Уравнение эллипса
    mask = ((x - x0)**2 / a**2 + (y - y0)**2 / b**2) <= 1
    
    # Отрисовка эллипса
    image[mask] = ellipse_color
    
    return image

def test_draw():
    # Тест прямоугольника
    rect = draw_rectangle(4, 3, 10, 10, [1, 0, 0], [0, 0, 0])
    assert rect.shape == (10, 10, 3)
    
    # Тест эллипса
    ellipse = draw_ellipse(3, 2, 10, 10, [0, 1, 0], [0, 0, 0])
    assert ellipse.shape == (10, 10, 3)
    
    print("Тесты задачи 6 пройдены")

# ===== ЗАДАЧА 7 =====
def analyze_time_series(series, window_size):
    # Основные статистики
    mean = np.mean(series)
    variance = np.var(series)
    std = np.std(series)
    
    # Локальные экстремумы
    local_maxima = []
    local_minima = []
    
    for i in range(1, len(series) - 1):
        if series[i] > series[i-1] and series[i] > series[i+1]:
            local_maxima.append(i)
        elif series[i] < series[i-1] and series[i] < series[i+1]:
            local_minima.append(i)
    
    # Скользящее среднее
    moving_avg = np.convolve(series, np.ones(window_size)/window_size, mode='valid')
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'local_maxima': np.array(local_maxima),
        'local_minima': np.array(local_minima),
        'moving_average': moving_avg
    }

def test_time_series():
    series = np.array([1, 3, 7, 1, 2, 6, 0, 5])
    result = analyze_time_series(series, 3)
    
    assert abs(result['mean'] - 3.125) < 1e-10
    assert abs(result['variance'] - 5.859375) < 1e-10
    assert np.array_equal(result['local_maxima'], np.array([2, 5]))
    assert np.array_equal(result['local_minima'], np.array([3, 6]))
    assert len(result['moving_average']) == len(series) - 3 + 1
    
    print("Тесты задачи 7 пройдены")

# ===== ЗАДАЧА 8 =====
def one_hot_encoding(labels):
    n_classes = len(np.unique(labels))
    encoding = np.zeros((len(labels), n_classes))
    
    for i, label in enumerate(labels):
        encoding[i, label] = 1
    
    return encoding

def test_one_hot():
    labels = np.array([0, 2, 3, 0])
    result = one_hot_encoding(labels)
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [1, 0, 0, 0]])
    assert np.array_equal(result, expected)
    
    # Тест с другими метками
    labels2 = np.array([1, 1, 0])
    result2 = one_hot_encoding(labels2)
    expected2 = np.array([[0, 1],
                          [0, 1],
                          [1, 0]])
    assert np.array_equal(result2, expected2)
    
    print("Тесты задачи 8 пройдены")

# Демонстрация работы функций
if __name__ == "__main__":
    # Запуск всех тестов
    test_sum_prod()
    test_binarize()
    test_unique()
    test_chess()
    test_draw()
    test_time_series()
    test_one_hot()
    
    # Демонстрация задачи 4
    print("\n=== Демонстрация задачи 4 ===")
    analyze_matrix(5, 5)
    
    # Демонстрация задачи 6
    print("\n=== Демонстрация задачи 6 ===")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    rectangle = draw_rectangle(6, 4, 10, 10, [1, 0, 0], [0.5, 0.5, 0.5])
    ax1.imshow(rectangle)
    ax1.set_title('Прямоугольник')
    
    ellipse = draw_ellipse(3, 2, 10, 10, [0, 1, 0], [0.5, 0.5, 0.5])
    ax2.imshow(ellipse)
    ax2.set_title('Эллипс')
    
    plt.show()
    
    print("Все тесты пройдены и демонстрации завершены!")