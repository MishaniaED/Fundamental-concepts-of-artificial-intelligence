import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def egg_holder(x):
    """Функция подставка для яиц (Egg Holder)."""
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(
        np.sqrt(np.abs(x[0] - (x[1] + 47))))


def egg_holder_gradient(x):
    """Градиент функции подставка для яиц (Egg Holder)."""
    gradient = np.zeros_like(x)
    sqrt_abs_term_1 = np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))
    sqrt_abs_term_2 = np.sqrt(np.abs(x[0] - (x[1] + 47)))

    gradient[0] = -0.5 * np.sign(x[0]) * np.cos(sqrt_abs_term_1) * np.sin(sqrt_abs_term_1) - np.sign(
        x[0] - (x[1] + 47)) * np.cos(sqrt_abs_term_2) * np.sin(sqrt_abs_term_2)
    gradient[1] = -np.cos(sqrt_abs_term_1) * np.sin(sqrt_abs_term_1) - np.sign(x[0] - (x[1] + 47)) * np.cos(
        sqrt_abs_term_2) * np.sin(sqrt_abs_term_2)

    return gradient


def rosenbrock(x):
    """Функция Розенброка."""
    return sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rosenbrock_gradient(x):
    """Градиент функции Розенброка."""
    gradient = np.zeros_like(x)
    gradient[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    gradient[1:-1] = 200 * (x[1:-1] - x[:-2] ** 2) - 400 * x[1:-1] * (x[2:] - x[1:-1] ** 2) - 2 * (1 - x[1:-1])
    gradient[-1] = 200 * (x[-1] - x[-2] ** 2)
    return gradient


def gradient_descent(learning_rate, max_iterations, initial_position, cost_function, gradient_function,
                     name_of_test_func):
    """Классический градиентный спуск с 3D визуализацией."""
    fig = go.Figure()
    if name_of_test_func == 'Подставка для яиц':
        # Генерация сетки для визуализации функции Подставка для яиц
        x = np.linspace(-2.0, 2.0, 100)
        y = np.linspace(-2.0, 2.0, 100)
        X, Y = np.meshgrid(x, y)
        Z = egg_holder(np.vstack([X.ravel(), Y.ravel()]))
    else:
        # Генерация сетки для визуализации функции Розенброка
        x = np.linspace(-2.0, 2.0, 100)
        y = np.linspace(-2.0, 2.0, 100)
        X, Y = np.meshgrid(x, y)
        Z = rosenbrock(np.vstack([X.ravel(), Y.ravel()]))

    fig.add_trace(go.Surface(x=X, y=Y, z=Z.reshape(X.shape), opacity=0.3, colorscale='viridis'))

    current_position = initial_position.astype(np.float64).copy()

    for i in range(max_iterations):
        gradient = gradient_function(current_position)
        current_position -= learning_rate * gradient

        cost = cost_function(current_position)

        # Визуализация текущей позиции на поверхности функции
        fig.add_trace(go.Scatter3d(x=[current_position[0]], y=[current_position[1]], z=[cost],
                                   mode='markers', marker=dict(color='red', size=4)))

        if i > 0:
            # Соединение точек линиями
            fig.add_trace(go.Scatter3d(x=[previous_position[0], current_position[0]],
                                       y=[previous_position[1], current_position[1]],
                                       z=[previous_cost, cost],
                                       mode='lines', line=dict(color='blue', width=2)))

        # Сохранение предыдущей позиции для следующей итерации
        previous_position = current_position.copy()
        previous_cost = cost

    # Обновляем макет графика
    fig.update_layout(scene=dict(aspectmode="cube"))
    fig.show()

    return current_position


# Параметры для градиентного спуска
learning_rate = 0.001
max_iterations = 100
initial_position = np.array([-1.8, -1.4])

# Запуск градиентного спуска для функции Розенброка с визуализацией
result_rosenbrock = gradient_descent(learning_rate, max_iterations, initial_position, rosenbrock, rosenbrock_gradient,
                                     'Розенброк')

print("\nРезультат оптимизации:")
print(f"Минимум функции Розенброка достигается в точке: {result_rosenbrock}")
print(f"Значение функции в минимуме: {rosenbrock(result_rosenbrock)}")

# Параметры для градиентного спуска
learning_rate = 0.01
max_iterations = 100
initial_position = np.array([200, 200])

# Запуск градиентного спуска для функции подставка для яиц (Egg Holder)
result_egg = gradient_descent(learning_rate, max_iterations, initial_position, egg_holder, egg_holder_gradient,
                              'Подставка для яиц')

print("\nРезультат оптимизации:")
print(f"Минимум функции Швефеля достигается в точке: {result_egg}")
print(f"Значение функции в минимуме: {egg_holder(result_egg)}")
