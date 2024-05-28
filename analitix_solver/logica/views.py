from django.shortcuts import render
from logica.oneVariable import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')

def Home(request):
    return render(request, "home.html")

def about_view(request):
    return render(request, 'about.html')

def help(request):
    return render(request, 'help.html')

def Methods(request):
    return render(request, "methods.html")

def false_position_method(request):
    data = ()
    if request.method == 'POST':
        fx = request.POST["function"]

        x0 = request.POST["a_value"]
        X0 = float(x0)

        xi = request.POST["b_value"]
        Xi = float(xi)

        tol = request.POST["tolerance"]
        Tol = float(tol)

        niter = request.POST["max_iterations"]
        Niter = int(niter)

        data = false_rule(X0, Xi, Niter, Tol, fx)
        try:
            # Convierte la cadena de texto de la función en una función de Sympy
            function = sympify(fx)
        except Exception as e:
            error_message = f'Error: {e}'
            return render(request, './oneVariable/false_position.html', {'error_message': error_message})

        # Define la función evaluada en un rango de valores
        x = np.linspace(-10, 10, 1000)
        y = np.array([function.evalf(subs={'x': xi}) for xi in x])

        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph of the Function')
        plt.grid(True)
        plt.savefig('static/img/graph.png')  
    if data:
        return render(request, './oneVariable/false_position.html', {'data': data})
    
    return render(request, "./oneVariable/false_position.html")

def fixed_point_method(request):
    data = ()
    if request.method == 'POST':
        fx = request.POST["function_x"]
        gx = request.POST["function_g"]

        x0 = request.POST["initial_value"]
        x0 = float(x0)

        tol = request.POST["tolerance"]
        Tol = float(tol)

        niter = request.POST["max_iterations"]
        niter = int(niter)

        data = fixed_point(x0,Tol,niter,fx,gx)
        try:
            # Convierte la cadena de texto de la función en una función de Sympy
            function = sympify(fx)
            y1 = sympify(gx)
        except Exception as e:
            error_message = f'Error: {e}'
            return render(request, './oneVariable/fixed_point.html', {'error_message': error_message})

        # Define la función evaluada en un rango de valores
        x = np.linspace(-10, 10, 1000)
        y = np.array([function.evalf(subs={'x': xi}) for xi in x])
        
        y1 = np.array([y1.evalf(subs={'x': xi}) for xi in x])

        plt.figure(figsize=(8, 6))

        plt.plot(x, y, label='F') 
        plt.plot(x, y1, label='F\'')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph of the Functions')
        plt.grid(True)
        plt.savefig('static/img/graph.png')

    if data:  
        return render(request, './oneVariable/fixed_point.html', {'data': data})

    return render(request, "./oneVariable/fixed_point.html")

def newton_method(request):
    data= ()
    if request.method == 'POST':
        fx = request.POST["function_f"]
        derf = request.POST["function_g"]

        x0 = request.POST["initial_value"]
        X0 = float(x0)

        tol = request.POST["tolerance"]
        Tol = float(tol)

        niter = request.POST["max_iterations"]
        Niter = int(niter)

        data = newton(X0, Tol, Niter, fx, derf)

        try:
            # Convierte la cadena de texto de la función en una función de Sympy
            function = sympify(fx)
            y1 = sympify(derf)
        except Exception as e:
            error_message = f'Error: {e}'
            return render(request, './oneVariable/newton.html', {'error_message': error_message})

        # Define la función evaluada en un rango de valores
        x = np.linspace(-10, 10, 1000)
        y = np.array([function.evalf(subs={'x': xi}) for xi in x])
        
        y1 = np.array([y1.evalf(subs={'x': xi}) for xi in x])

        plt.figure(figsize=(8, 6))

        plt.plot(x, y, label='F') 
        plt.plot(x, y1, label='F\'')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph of the Functions')
        plt.grid(True)
        plt.savefig('static/img/graph.png')

    if data: 
        return render(request, './oneVariable/newton.html', {'data': data})
    
    return render(request, "./oneVariable/newton.html")

def secant_method(request):
    data= ()
    if request.method == 'POST':
        fx = request.POST["function"]
        tol = request.POST["tolerance"]
        Tol = float(tol)
        niter = request.POST["max_iterations"]
        Niter = int(niter)
        X0 = request.POST["a_value"]
        x0 = float(X0)
        X1 = request.POST["b_value"]
        x1 = float(X1)

        data = secant(fx, Tol, Niter, x0, x1)
        try:
            # Convierte la cadena de texto de la función en una función de Sympy
            function = sympify(fx)
        except Exception as e:
            error_message = f'Error: {e}'
            return render(request, './oneVariable/secant.html', {'error_message': error_message})

        # Define la función evaluada en un rango de valores
        x = np.linspace(-10, 10, 1000)
        y = np.array([function.evalf(subs={'x': xi}) for xi in x])

        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph of the Function')
        plt.grid(True)
        plt.savefig('static/img/graph.png') 

    if data:
        return render(request, './oneVariable/secant.html', {'data': data})

    return render(request, "./oneVariable/secant.html")
'''

'''

def multiple_roots_method(request):
    data = ()
    if request.method == 'POST':
        fx = request.POST["function_f"]
        ffx = request.POST["function_ff"]
        fffx = request.POST["function_fff"]
        Tol = request.POST["tolerance"]
        niter = request.POST["max_iterations"]
        x0 = request.POST["initial_value"]
        try:
            
            x0 = float(x0)
            Niter = int(niter)
            Tol = float(Tol)
            data = multiple_roots(fx,ffx,fffx,x0,Tol,Niter)
        
            # Convierte la cadena de texto de la función en una función de Sympy
            function = sympify(fx)
            function1 = sympify(ffx)
            function2 = sympify(fffx)
            
        except Exception as e:
            error_message = f'Error: {e}'
            return render(request, './oneVariable/multiple_roots.html', {'error_message': error_message})

        # Define la función evaluada en un rango de valores
        x = np.linspace(-10, 10, 1000)

        y = np.array([function.evalf(subs={'x': xi}) for xi in x])
        y1 = np.array([function1.evalf(subs={'x': xi}) for xi in x])
        y2 = np.array([function2.evalf(subs={'x': xi}) for xi in x])

        plt.figure(figsize=(8, 6))
        
        plt.plot(x, y,label="f")
        plt.plot(x, y1,label="f\'")
        plt.plot(x, y2, label="f\'\'")

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph of the Function')
        plt.grid(True)
        plt.savefig('static/img/graph.png') 

    if data:
        return render(request, "./oneVariable/multiple_roots.html", {"data":data})
    return render(request, "./oneVariable/multiple_roots.html")


'''

'''
def bisection_method(request):
    data = ()
    if request.method == 'POST':
        fx = request.POST["function"]
        tol = request.POST["tolerance"]
        Tol = float(tol)
        niter = request.POST["max_iterations"]
        Niter = int(niter)
        
        a_value = request.POST["a_value"]
        a_value = float(a_value)
        b_value = request.POST["b_value"]
        b_value = float(b_value)

        data = bisection(fx, Tol, Niter, a_value, b_value)

        try:
            # Convierte la cadena de texto de la función en una función de Sympy
            function = sympify(fx)
        except Exception as e:
            error_message = f'Error: {e}'
            return render(request, './oneVariable/bisection.html', {'error_message': error_message})

        # Define la función evaluada en un rango de valores
        x = np.linspace(-10, 10, 1000)
        y = np.array([function.evalf(subs={'x': xi}) for xi in x])

        plt.figure(figsize=(8, 6))
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Graph of the Function')
        plt.grid(True)
        plt.savefig('static/img/graph.png')  

    if data:
        return render(request, './oneVariable/bisection.html', {'data': data})

    return render(request, './oneVariable/bisection.html')

'''

'''
def sor_method(request):
    context = {}
    if request.method == 'POST':
        size = int(request.POST['matrix-size'])
        A = np.zeros((size, size), dtype=float)
        b = np.zeros(size, dtype=float)
        for i in range(size):
            for j in range(size):
                A[i, j] = float(request.POST[f'matrixA{i}{j}'])
            b[i] = float(request.POST[f'vectorB{i}'])
        initial_guess = [float(x) for x in request.POST.get('initial_guess', '[0]*size').strip('[]').split(',')]
        omega = float(request.POST['omega'])
        tolerance = float(request.POST['tolerance'])
        max_iterations = int(request.POST['max_iterations'])
        solution, iterations, iter_details, spectral_radius = sor_solver(A, b, omega, initial_guess, tolerance, max_iterations)
        context.update({
            'solution': solution.tolist() if solution is not None else 'No solution found',
            'iterations': iterations,
            'iter_details': iter_details,
            'max_iterations': max_iterations,
            'columns': ['iter', 'x values', 'E'],
            'spectral_radius': spectral_radius
        })
    return render(request, 'oneVariable/sor.html', context)
'''

'''
def jacobi_method(request):
    context = {}
    if request.method == 'POST':
        size = int(request.POST['matrix-size'])
        A = np.zeros((size, size), dtype=float)
        b = np.zeros(size, dtype=float)
        for i in range(size):
            for j in range(size):
                A[i, j] = float(request.POST[f'matrixA{i}{j}'])
            b[i] = float(request.POST[f'vectorB{i}'])
        initial_guess = [float(x) for x in request.POST.get('initial_guess', '[0]' * size).strip('[]').split(',')]
        tolerance = float(request.POST['tolerance'])
        max_iterations = int(request.POST['max_iterations'])
        solution, iterations, iter_details, spectral_radius = jacobi_solver(A, b, initial_guess, tolerance, max_iterations)
        context.update({
            'solution': solution.tolist() if solution is not None else 'No solution found',
            'iterations': iterations,
            'iter_details': iter_details,
            'max_iterations': max_iterations,
            'columns': ['iter', 'x values', 'E'],
            'spectral_radius': spectral_radius
        })
    return render(request, 'oneVariable/jacobi.html', context)
'''

'''
def gauss_method(request):
    context = {}
    if request.method == 'POST':
        size = int(request.POST['matrix-size'])
        A = np.zeros((size, size), dtype=float)
        b = np.zeros(size, dtype=float)
        for i in range(size):
            for j in range(size):
                A[i, j] = float(request.POST[f'matrixA{i}{j}'])
            b[i] = float(request.POST[f'vectorB{i}'])
        initial_guess = [float(x) for x in request.POST.get('initial_guess', '[0]' * size).strip('[]').split(',')]
        tolerance = float(request.POST['tolerance'])
        max_iterations = int(request.POST['max_iterations'])
        solution, iterations, iter_details, spectral_radius = gauss_seidel_solver(A, b, initial_guess, tolerance, max_iterations)
        context.update({
            'solution': solution.tolist() if solution is not None else 'No solution found',
            'iterations': iterations,
            'iter_details': iter_details,
            'max_iterations': max_iterations,
            'columns': ['iter', 'x values', 'E'],
            'spectral_radius': spectral_radius
        })
    return render(request, 'oneVariable/gauss_seidel.html', context)
"""

"""
def vandermonde_method(request):
    context = {}
    if request.method == 'POST':
        try:
            x_values = request.POST.get('x_values', '')
            y_values = request.POST.get('y_values', '')
            x = list(map(float, x_values.split(',')))
            y = list(map(float, y_values.split(',')))
            V, coeffs, polynomial, formatted_polynomial = vandermonde_interpolation(x, y)
            context['matrix'] = V.tolist()
            context['coefficients'] = coeffs.tolist()
            context['polynomial'] = formatted_polynomial
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, color='red', label='Data Points')
            xs = np.linspace(min(x), max(x), 500)
            ys = np.polyval(coeffs[::-1], xs)
            plt.plot(xs, ys, label=f'Interpolating polynomial: {formatted_polynomial}')
            plt.title('Graph of the Vandermonde Interpolation')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.savefig('static/img/graph.png')
            context['graph'] = 'img/graph.png'
        except Exception as e:
            context['error'] = str(e)
    return render(request, 'oneVariable/vandermonde.html', context)

"""

"""
def newton_interpolation_method(request):
    context = {}
    if request.method == 'POST':
        x_values = request.POST.get('x_values', '').strip()
        y_values = request.POST.get('y_values', '').strip()
        try:
            x = list(map(float, x_values.split(',')))
            y = list(map(float, y_values.split(',')))
            if len(x) != len(y):
                raise ValueError("The number of X values must match the number of Y values.")
            if len(set(x)) != len(x):
                raise ValueError("X values must be distinct.")
            if len(set(y)) != len(y):
                raise ValueError("Y values must be distinct.")
            polynomial, diff_table, coefficients = newton_interpolation(x, y)
            table_data = []
            for i in range(len(x)):
                row = {'index': i, 'x': x[i], 'y': diff_table[i, 0], 'diffs': diff_table[i, 1:].tolist()}
                table_data.append(row)
            x_plot = np.linspace(min(x) - 1, max(x) + 1, 400)
            y_plot = [polynomial.subs('x', xi).evalf() for xi in x_plot]
            plt.figure(figsize=(8, 6))
            plt.plot(x_plot, y_plot, label='Newton Polynomial', color='blue')
            plt.scatter(x, y, color='red', label='Data Points')
            plt.title('Newton Interpolation Polynomial')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.savefig('static/img/graph.png')
            plt.close()
            context['polynomial'] = str(polynomial)
            context['coefficients'] = coefficients
            context['graph'] = 'img/graph.png'
            context['table_data'] = table_data
            context['range'] = range(1, len(x))
        except ValueError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = "An unexpected error occurred: {}".format(e)
    return render(request, 'oneVariable/newton_interpolation.html', context)
"""

"""
def lagrange_interpolation_method(request):
    context = {}
    if request.method == 'POST':
        x_values = request.POST.get('x_values', '').strip()
        y_values = request.POST.get('y_values', '').strip()
        try:
            x = list(map(float, x_values.split(',')))
            y = list(map(float, y_values.split(',')))
            if len(x) != len(y):
                raise ValueError("The number of X values must match the number of Y values.")
            if len(set(x)) != len(x):
                raise ValueError("X values must be distinct.")
            if len(set(y)) != len(y):
                raise ValueError("Y values must be distinct.")
            polynomial, Li_expr = lagrange_interpolation(x, y)
            x_plot = np.linspace(min(x) - 1, max(x) + 1, 400)
            y_plot = [polynomial.subs('x', xi).evalf() for xi in x_plot]
            plt.figure(figsize=(8, 6))
            plt.plot(x_plot, y_plot, label='Lagrange Polynomial', color='blue')
            plt.scatter(x, y, color='red', label='Data Points')
            plt.title('Lagrange Interpolation Polynomial')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.savefig('static/img/graph.png')
            plt.close()
            
            context['polynomial'] = str(polynomial)
            context['Li_expr'] = Li_expr
            context['x_values'] = x
            context['y_values'] = y
            context['graph'] = 'img/graph.png'
            
        except ValueError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = "An unexpected error occurred: {}".format(e)

    return render(request, 'oneVariable/lagrange.html', context)

"""

"""

def spline_method(request):
    context = {}
    if request.method == 'POST':
        x_values = request.POST.get('x_values').split(',')
        y_values = request.POST.get('y_values').split(',')
        degree = int(request.POST.get('degree'))
        
        x_values = [float(i) for i in x_values]
        y_values = [float(i) for i in y_values]

        coefficients, polynomials = spline_interpolation(x_values, y_values, degree)
        x_plot = np.linspace(min(x_values), max(x_values), 400)
        y_plots = []

        for poly in polynomials:
            p = np.poly1d(poly)
            y_plots.append(p(x_plot))

        plt.figure(figsize=(10, 5))
        for y_plot in y_plots:
            plt.plot(x_plot, y_plot)
        plt.scatter(x_values, y_values, color='red', zorder=5)
        plt.title('Spline Interpolation')
        plt.grid(True)
        plt.legend()
        plt.savefig('static/img/graph.png')
        plt.close()

        context['coefficients'] = coefficients
        context['polynomials'] = [(i, format_polynomial(poly, degree)) for i, poly in enumerate(polynomials)]
        context['graph'] = 'img/graph.png'
        context['degree'] = degree

    return render(request, 'oneVariable/spline.html', context)

def format_polynomial(coefficients, degree):
    if degree == 1:
        return f"{coefficients[0]:.12f}x + {coefficients[1]:.12f}"
    elif degree == 2:
        return f"{coefficients[0]:.12f}x^2 + {coefficients[1]:.12f}x + {coefficients[2]:.12f}"
    elif degree == 3:
        return f"{coefficients[0]:.12f}x^3 + {coefficients[1]:.12f}x^2 + {coefficients[2]:.12f}x + {coefficients[3]:.12f}"