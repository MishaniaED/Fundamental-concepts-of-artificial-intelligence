import numpy as np
import plotly.graph_objects as go
import sympy as sp


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
        x = np.linspace(-150.0, 150.0, 300)
        y = np.linspace(-150.0, 200.0, 300)
        X, Y = np.meshgrid(x, y)
        Z = egg_holder(np.vstack([X.ravel(), Y.ravel()]))
    else:
        # Генерация сетки для визуализации функции Розенброка
        x = np.linspace(-3.0, 3.0, 100)
        y = np.linspace(-3.0, 10.0, 100)
        X, Y = np.meshgrid(x, y)
        Z = rosenbrock(np.vstack([X.ravel(), Y.ravel()]))

    fig.add_trace(go.Surface(x=X, y=Y, z=Z.reshape(X.shape), opacity=0.3, colorscale='Jet'))

    current_position = initial_position.astype(np.float64).copy()

    for i in range(max_iterations+1):
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


def analytical_solution(name_of_test_func):
    """Вычисление аналитического решения для заданных тестовых функций."""
    if name_of_test_func == 'Подставка для яиц': # Возможно придется поменять функцию

        return np.array([0.0, 0.0])  # Пример аналитического решения для подставки для яиц
    elif name_of_test_func == 'Розенброк':
        x, y = sp.symbols('x y')
        f = 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

        # Вычисляем градиент
        gradient = [sp.diff(f, var) for var in (x, y)]

        # Решаем систему уравнений, приравнивая градиент к нулю
        solution = sp.solve(gradient, (x, y))

        return solution
    else:
        return None


def compute_error(found_solution, analytical_solution):
    """Вычисление погрешности между найденным и аналитическим решениями."""
    if analytical_solution is not None:
        error = np.linalg.norm(found_solution - analytical_solution)
        return error
    else:
        return None


def optimization_pipeline(learning_rate, max_iterations, initial_position, cost_function, gradient_function,
                          name_of_test_func):
    """Пайплайн тестирования алгоритма оптимизации."""

    # Запуск градиентного спуска
    found_solution = gradient_descent(learning_rate, max_iterations, initial_position, cost_function, gradient_function,
                                      name_of_test_func)

    # Вычисление аналитического решения
    analytical_solution_point = analytical_solution(name_of_test_func)

    # Вычисление погрешности
    # error = compute_error(found_solution, analytical_solution_point)

    # Вывод результатов
    print("\nРезультат оптимизации:")
    print(f'Минимум функции достигается в точке: {found_solution}')
    print(
        f"Значение функции в минимуме: {egg_holder(found_solution) if name_of_test_func == 'Подставка для яиц' else rosenbrock(found_solution)}")
    if analytical_solution_point is not None:
        print(f"Аналитическое решение: {analytical_solution_point}")
        # print(f"Погрешность: {error}")


if __name__ == "__main__":
    # Параметры для градиентного спуска
    learning_rate = 0.0001
    max_iterations = 100
    initial_position = np.array([-1.8, -1.4])

    # Запуск градиентного спуска для функции Розенброка с визуализацией
    optimization_pipeline(learning_rate, max_iterations, initial_position, rosenbrock, rosenbrock_gradient, 'Розенброк')

    # Параметры для градиентного спуска
    learning_rate = 50
    max_iterations = 100
    initial_position = np.array([16, 59])

    # Запуск градиентного спуска для функции подставка для яиц (Egg Holder), из-за частых перегибов не удается найти
    optimization_pipeline(learning_rate, max_iterations, initial_position, egg_holder, egg_holder_gradient,
                          'Подставка для яиц')
