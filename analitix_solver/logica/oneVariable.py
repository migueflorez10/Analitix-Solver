import sympy as sympy
from sympy import Symbol, sympify, Abs, diff
from sympy.abc import x
from sympy import symbols, sympify, N
import numpy as np
from . import convertTable 

'''
This code implements the bisection method to find a root of a function f(x) on a given interval [a,b]. 
'''
def bisection(fx, Tol, Niter, a, b):
    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "iterations": Niter,
        "errors": list()
    }

    # Configuraciones iniciales
    datos = list()
    x = Symbol('x')
    i = 1
    error = 1.0000000
    Fun = sympify(fx)

    Fa = Fun.subs(x, a) # Funcion evaluada en a
    Fa = Fa.evalf()

    xm0 = 0.0
    Fxm = 0

    xm = (a + b)/2 # Punto intermedio

    Fxm = Fun.subs(x, xm) # Funcion evaluada en Xm
    Fxm = Fxm.evalf()
    
    try:
        # Datos con formato dado
        datos.append([0, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fxm)])
        # Se repite hasta que el intervalo sea lo pequeÃ±o que se desee
        while (error > Tol) and (i < Niter): 
            # Se elecciona un intervalo inicial, donde el valor de la funcion cambie de signo en [a,b]
            if (Fa*Fxm < 0):
                b = xm
            else:
                # Cambia de signo en [m,b]
                a = xm 

            xm0 = xm
            # Se calcula el punto intermedio del intervalo - Divide el intervalo a la mitadd
            xm = (a+b)/2 
            Fxm = Fun.subs(x, xm)
            # Se evalua el punto intermedio en la funcion
            Fxm = Fxm.evalf() 
            # Se calcula el error
            error = Abs(xm-xm0) 
            # Se van agregando las soluciones con el formato deseado
            datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), 
                            '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fxm), '{:^15.7E}'.format(error)]) 

            i += 1
        
        
            
            
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = datos
    output["root"] = xm
    convertTable.tableToText(output["columns"], output["results"], "bisection")
    return output

'''

'''
def fixed_point(X0, Tol, Niter, fx, gx):

    output = {
        "columns": ["iter", "xi", "g(xi)", "f(xi)", "E"],
        "iterations": Niter,
        "errors": list()
    }

    #configuración inicial
    datos = list()
    x = sympy.Symbol('x')
    i = 1
    Tol
    error = 1.000

    Fx = sympify(fx)
    Gx = sympify(gx)

    # Iteracion 0
    xP = X0 # Valor inicial (Punto evaluacion)
    xA = 0.0

    Fa = Fx.subs(x, xP) # Funcion evaluada en el valor inicial
    Fa = Fa.evalf()

    Ga = Gx.subs(x, xP) # Funcion G evaluada en el valor inicial
    Ga = Ga.evalf()

    datos.append([0, '{:^15.7f}'.format(float(xA)), '{:^15.7f}'.format(
        float(Ga)), '{:^15.7E}'.format(float(Fa))])
    try:
        while((error > Tol) and (i < Niter)): # Se repite hasta que el error sea menor a la tolerancia

            # Se evalua el valor inicial en G, para posteriormente evaluar este valor en la funcion F siendo-> Xn=G(x) y F(xn) = F(G(x))
            Ga = Gx.subs(x, xP) # Funcion G evaluada en el punto de inicial
            xA = Ga.evalf()

            Fa = Fx.subs(x, xA)# Funcion evaluada en el valor de la evalucacion de G
            Fa = Fa.evalf()

            error = Abs(xA - (xP)) # Se calcula el error 

            xP = xA # Nuevo punto de evaluacion (Punto inicial)

            datos.append([i, '{:^15.7f}'.format(float(xA)), '{:^15.7f}'.format(
                float(Ga)), '{:^15.7E}'.format(float(Fa)), '{:^15.7E}'.format(float(error))])

            i += 1

    except BaseException as e:
        output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xA
    convertTable.tableToText(output["columns"], output["results"], "fixed_point")
    return output

"""

"""
def false_rule(a, b, Niter, Tol, fx):
    
    output = {
        "columns": ["iter", "a", "xm", "b", "f(xm)", "E"],
        "iterations": Niter,
        "errors": list()
    }

    #configuración inicial
    datos = list()
    x = sympy.Symbol('x')
    i = 1
    cond = Tol
    error = 1.0000000

    Fun = sympify(fx)

    xm = 0
    xm0 = 0
    Fx_2 = 0
    Fx_3 = 0
    Fa = 0
    Fb = 0

    try:
        while (error > cond) and (i < Niter):
            if i == 1:
                Fx_2 = Fun.subs(x, a)
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b)
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb*a - Fa*b)/(Fb-Fa)
                Fx_3 = Fun.subs(x, xm)
                Fx_3 = Fx_3.evalf()
                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3)])
            else:

                if (Fa*Fx_3 < 0):
                    b = xm
                else:
                    a = xm

                xm0 = xm
                Fx_2 = Fun.subs(x, a) #Función evaluada en a
                Fx_2 = Fx_2.evalf()
                Fa = Fx_2

                Fx_2 = Fun.subs(x, b) #Función evaluada en a
                Fx_2 = Fx_2.evalf()
                Fb = Fx_2

                xm = (Fb*a - Fa*b)/(Fb-Fa) #Calcular intersección en la recta en el eje x

                Fx_3 = Fun.subs(x, xm) #Función evaluada en xm (f(xm))
                Fx_3 = Fx_3.evalf()

                error = Abs(xm-xm0)
                er = sympify(error)
                error = er.evalf()
                datos.append([i, '{:^15.7f}'.format(a), '{:^15.7f}'.format(xm), '{:^15.7f}'.format(b), '{:^15.7E}'.format(Fx_3), '{:^15.7E}'.format(error)])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = datos
    output["root"] = xm
    convertTable.tableToText(output["columns"], output["results"], "false_position")
    return output

'''

'''
def newton(x0, Tol, Niter, fx, df):

    output = {
        "columns": ["N", "xi", "F(xi)", "E"],
        "errors": list()
    }

    #configuración inicial
    datos = list()
    x = sympy.Symbol('x')
    Fun = sympify(fx)
    DerF = sympify(df)


    xn = []
    derf = []
    xi = x0 # Punto de inicio
    f = Fun.evalf(subs={x: x0}) #función evaluada en x0
    derivada = DerF.evalf(subs={x: x0}) #función derivada evaluada en x0
    c = 0
    Error = 100
    xn.append(xi)

    try:
        datos.append([c, '{:^15.7f}'.format(x0), '{:^15.7f}'.format(f)])

        # Al evaluar la derivada en el punto inicial,se busca que sea diferente de 0, ya que al serlo nos encontramos en un punto de inflexion
        #(No se puede continuar ya que la tangente es horinzontal)
        while Error > Tol and f != 0 and derivada != 0 and c < Niter: # El algoritmo converge o se alcanzo limite de iteraciones fijado

            xi = xi-f/derivada # Estimacion del siguiente punto aproximado a la raiz (nuevo valor inicial)
            derivada = DerF.evalf(subs={x: xi}) # Evaluacion de la derivada con el nuevo valor inicial (xi)
            f = Fun.evalf(subs={x: xi}) # Evaluacion de la derivada con el nuevo valor inicial (xi)
            xn.append(xi)
            c = c+1
            Error = abs(xn[c]-xn[c-1]) # Se reduce entre cada iteracion (Representado por el tramo)
            derf.append(derivada)
            datos.append([c, '{:^15.7f}'.format(float(xi)), '{:^15.7E}'.format(
                float(f)), '{:^15.7E}'.format(float(Error))])

    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))
        return output

    output["results"] = datos
    output["root"] = xi
    convertTable.tableToText(output["columns"], output["results"], "newton")
    return output

'''

'''
def secant(fx, tol, Niter, x0, x1):
    
    output = {
        "columns": ["iter", "xi", "f(xi)", "E"],
        "errors": list()
    }

    results = list()
    x = Symbol('x')
    i = 0
    cond = tol
    error = 1.0000000

    Fun = sympify(fx)

    y = x0
    Fx0 = Fun
    Fx1 = Fun

    try:
        while((error > cond) and (i < Niter)): #criterios de parada
            if i == 0:
                Fx0 = Fun.subs(x, x0) #Evaluacion en el valor inicial X0
                Fx0 = Fx0.evalf()
                results.append([i, '{:^15.7f}'.format(float(x0)), '{:^15.7E}'.format(float(Fx0))])
            elif i == 1:
                Fx1 = Fun.subs(x, x1)#Evaluacion en el valor inicial X1
                Fx1 = Fx1.evalf()
                results.append([i, '{:^15.7f}'.format(float(x1)), '{:^15.7E}'.format(float(Fx1))])
            else:
                y = x1 
                # Se calcula la secante
                x1 = x1 - (Fx1*(x1 - x0)/(Fx1 - Fx0)) # Punto de corte del intervalo usando la raiz de la secante, (xi+1)
                x0 = y

                Fx0 = Fun.subs(x, x0) #Evaluacion en el valor inicial X0
                Fx0 = Fx1.evalf() 

                Fx1 = Fun.subs(x, x1)#Evaluacion en el valor inicial X1
                Fx1 = Fx1.evalf()

                error = Abs(x1 - x0) # Tramo

                results.append([i, '{:^15.7f}'.format(float(x1)), '{:^15.7E}'.format(float(Fx1)), '{:^15.7E}'.format(float(error))])
            i += 1
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

        return output

    output["results"] = results
    output["root"] = y
    convertTable.tableToText(output["columns"], output["results"], "secant")
    return output

'''

'''

def multiple_roots(funct, first_derivative, second_derivative, x0, tol, max_count):
    x = symbols('x')
    f = sympify(funct)
    f_prime = sympify(first_derivative)
    f_double_prime = sympify(second_derivative)

    # Lista para almacenar los datos de cada iteración
    iterations_data = []
    errors = []

    x0 = float(x0)
    f_x = f.subs(x, x0)
    f_x_prime = f_prime.subs(x, x0)
    f_x_double_prime = f_double_prime.subs(x, x0)
    err = tol + 1
    cont = 0

    # Almacenar los datos de la primera iteración
    iterations_data.append([cont, "{:.10e}".format(N(x0)), "{:.2e}".format(float(N(f_x)))])
    errors.append("")
    try:
        while err > tol and cont < max_count:
            d = f_x_prime**2 - f_x * f_x_double_prime
            if d == 0:
                break
            
            x_ev = x0 - (f_x * f_x_prime) / d
            if x_ev == float('inf'):
                raise ValueError(f"Infinity value in step {cont}")

            f_x = f.subs(x, x_ev)
            f_x_prime = f_prime.subs(x, x_ev)
            f_x_double_prime = f_double_prime.subs(x, x_ev)
            err = abs(x_ev - x0)
            x0 = x_ev
            cont += 1

            iterations_data.append([cont, "{:.10e}".format(N(x0)), "{:.2e}".format(float(N(f_x))), "{:.2e}".format(float(N(err))) ])
        
    except BaseException as e:
        if str(e) == "can't convert complex to float":
            output["errors"].append(
                "Error in data: found complex in calculations")
        else:
            output["errors"].append("Error in data: " + str(e))

    output = {
        "columns": ["iter", "xm", "f(xm)", "E"],
        "iterations": iterations_data,
        "errors": errors
    }
    output["root"] = x_ev

    if f_x == 0:
        output['conclusion'] = f"The root was found for x = {x0:.15f}"
    elif err > tol:
        output['conclusion'] = f"An approximation of the root was found for x = {x0:.15f}, but the desired tolerance was not achieved."
    elif cont >= max_count:
        output['conclusion'] = "Given the number of iterations and the tolerance, it was impossible to find a satisfying root."
    else:
        output['conclusion'] = "The method exploded."
        
    convertTable.tableToText(output["columns"], output["iterations"], "multiple_roots")
    
    return output
    
'''

'''
def sor_solver(A, b, omega, initial_guess, tolerance, max_iterations):
    output = {}
    headers = ["iteration", "x_values", "error"]
    x = np.array(initial_guess)
    n = len(b)
    iter_details = []
    details = []
    
    x_new = np.copy(x)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    T = np.linalg.inv(D + omega * L).dot((1 - omega) * D - omega * U)
    spectral_radius = max(abs(np.linalg.eigvals(T)))
    for iteration in range(max_iterations):
        x_old = np.copy(x_new)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x_new[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
            x_new[i] = (1 - omega) * x_old[i] + omega * x_new[i]
        error = np.linalg.norm(x_new - x_old, ord=np.inf)
        formatted_error = f"{error:^15.7E}"
        iter_details.append({'iteration': iteration, 'x_values': np.round(x_new, 6).tolist(), 'error': formatted_error})
        
        details.append([iteration, np.round(x_new, 6).tolist(), formatted_error])
        
        if error < tolerance:
            break
        x = x_new
        
    output["results"] = details
    convertTable.tableToText(headers , output["results"] , "sor_solver")
    
    return np.round(x, 4), iteration + 1, iter_details, np.round(spectral_radius, 6)
'''

'''
def jacobi_solver(A, b, initial_guess, tolerance, max_iterations):
    x = np.array(initial_guess)
    n = len(b)
    iter_details = []
    x_new = np.copy(x)
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(D)
    T = -D_inv.dot(R)
    spectral_radius = max(abs(np.linalg.eigvals(T)))
    for iteration in range(max_iterations):
        x_old = np.copy(x)
        x_new = np.linalg.inv(D).dot(b - np.dot(R, x_old))
        error = np.linalg.norm(x_new - x_old, np.inf)
        formatted_error = f"{error:^15.7E}"
        iter_details.append({'iteration': iteration, 'x_values': np.round(x_new, 6).tolist(), 'error': formatted_error})
        if error < tolerance:
            break
        x = x_new
    return np.round(x, 4), iteration + 1, iter_details, np.round(spectral_radius, 6)
'''

'''

def gauss_seidel_solver(A, b, initial_guess, tolerance, max_iterations):
    x = np.array(initial_guess)
    n = len(b)
    iter_details = []
    x_new = np.copy(x)
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    T = np.linalg.inv(D + L).dot(-U)
    spectral_radius = max(abs(np.linalg.eigvals(T)))
    for iteration in range(max_iterations):
        x_old = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        error = np.linalg.norm(x_new - x_old, ord=np.inf)
        formatted_error = f"{error:^15.7E}"
        iter_details.append({'iteration': iteration, 'x_values': np.round(x_new, 6).tolist(), 'error': formatted_error})
        if error < tolerance:
            break
        x = x_new
    return np.round(x, 4), iteration + 1, iter_details, np.round(spectral_radius, 6)
"""

"""

def vandermonde_interpolation(x_values, y_values):
    try:
        x = np.array(x_values)
        y = np.array(y_values)
        if len(x) != len(y):
            raise ValueError("The number of x values must match the number of y values.")
        if len(set(x)) != len(x):
            raise ValueError("X values must be distinct.")
        if len(set(y)) != len(y):
            raise ValueError("Y values must be distinct.")
        V = np.vander(x, increasing=False)
        coeffs = np.linalg.solve(V, y)
        x_sym = sympy.symbols('x')
        polynomial = sum(sympy.N(coeff, 6) * x_sym**i for i, coeff in enumerate(coeffs[::-1]))
        polynomial_str = sympy.sstr(polynomial, full_prec=False)
        terms = polynomial_str.replace('**', '^').split(' + ')
        formatted_terms = []
        for term in terms:
            if 'x' in term:
                parts = term.split('*')
                if len(parts) == 2:
                    coeff, power = parts
                    if power == 'x^1':
                        power = 'x'
                    formatted_term = f"{sympy.N(coeff, 6)}*{power}"
                else:
                    formatted_term = term
            else:
                formatted_term = f"{sympy.N(term, 6)}"
            formatted_terms.append(formatted_term)
        formatted_polynomial = ' + '.join(formatted_terms).replace('+ -', '- ')
        return V, np.round(coeffs, 6), polynomial, formatted_polynomial
    except Exception as e:
        raise e
    
"""

"""
def newton_interpolation(x_values, y_values):
    n = len(x_values)
    x = sympy.symbols('x')
    diff_table = np.zeros((n, n), dtype=float)
    diff_table[:, 0] = y_values
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = (diff_table[i + 1][j - 1] - diff_table[i][j - 1]) / (x_values[i + j] - x_values[i])
    coefficients = diff_table[0, :n]
    polynomial = coefficients[0]
    term = sympy.S(1)
    for i in range(1, n):
        term *= (x - x_values[i-1])
        polynomial += coefficients[i] * term
    polynomial = sympy.simplify(polynomial)
    return polynomial, diff_table, np.round(coefficients.tolist(), 6)

"""


"""
def lagrange_interpolation(x_values, y_values):
    x = sympy.symbols('x')
    n = len(x_values)
    L = []
    Li_expr = []
    for i in range(n):
        li = 1
        numerators = []
        denominators = []
        for j in range(n):
            if i != j:
                li *= (x - x_values[j]) / (x_values[i] - x_values[j])
                num = f"(x - {x_values[j]:g})"
                denom = f"({x_values[i]:g} - {x_values[j]:g})"
                numerators.append(num)
                denominators.append(denom)
                numerator_expr = " * ".join(numerators)
        denominator_expr = " * ".join(denominators)
        full_expr = f"({numerator_expr}) / ({denominator_expr})"
        L.append(li)
        Li_expr.append((i, full_expr))

    polynomial = sum(y_values[i] * L[i] for i in range(n))
    polynomial = sympy.simplify(polynomial)
    return polynomial, Li_expr

"""

"""
def spline_interpolation(x_values, y_values, degree):
    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    n = len(x)
    m = (degree + 1) * (n - 1)
    A = np.zeros((m, m))
    b = np.zeros(m)
    coefficients = []
    polynomials = []

    if degree == 1:
        c = 0
        for i in range(n - 1):
            A[i, c] = x[i]
            A[i, c + 1] = 1
            b[i] = y[i]
            c += 2

        c = 0
        for i in range(1, n):
            A[n - 1 + i - 1, c] = x[i]
            A[n - 1 + i - 1, c + 1] = 1
            b[n - 1 + i - 1] = y[i]
            c += 2

    elif degree == 2:
        c = 0
        for i in range(n - 1):
            A[i, c] = x[i]**2
            A[i, c + 1] = x[i]
            A[i, c + 2] = 1
            b[i] = y[i]
            c += 3

        c = 0
        for i in range(1, n):
            A[n - 1 + i - 1, c] = x[i]**2
            A[n - 1 + i - 1, c + 1] = x[i]
            A[n - 1 + i - 1, c + 2] = 1
            b[n - 1 + i - 1] = y[i]
            c += 3

        c = 0
        for i in range(1, n - 1):
            A[2 * (n - 1) + i - 1, c] = 2 * x[i]
            A[2 * (n - 1) + i - 1, c + 1] = 1
            A[2 * (n - 1) + i - 1, c + 3] = -2 * x[i]
            A[2 * (n - 1) + i - 1, c + 4] = -1
            b[2 * (n - 1) + i - 1] = 0
            c += 3

        A[-1, 0] = 2
        b[-1] = 0

    elif degree == 3:
        c = 0
        for i in range(n - 1):
            A[i, c] = x[i]**3
            A[i, c + 1] = x[i]**2
            A[i, c + 2] = x[i]
            A[i, c + 3] = 1
            b[i] = y[i]
            c += 4

        c = 0
        for i in range(1, n):
            A[n - 1 + i - 1, c] = x[i]**3
            A[n - 1 + i - 1, c + 1] = x[i]**2
            A[n - 1 + i - 1, c + 2] = x[i]
            A[n - 1 + i - 1, c + 3] = 1
            b[n - 1 + i - 1] = y[i]
            c += 4

        c = 0
        for i in range(1, n - 1):
            A[2 * (n - 1) + 2 * (i - 1), c] = 3 * x[i]**2
            A[2 * (n - 1) + 2 * (i - 1), c + 1] = 2 * x[i]
            A[2 * (n - 1) + 2 * (i - 1), c + 2] = 1
            A[2 * (n - 1) + 2 * (i - 1), c + 4] = -3 * x[i]**2
            A[2 * (n - 1) + 2 * (i - 1), c + 5] = -2 * x[i]
            A[2 * (n - 1) + 2 * (i - 1), c + 6] = -1
            b[2 * (n - 1) + 2 * (i - 1)] = 0
            c += 4

        c = 0
        for i in range(1, n - 1):
            A[2 * (n - 1) + 2 * (i - 1) + 1, c] = 6 * x[i]
            A[2 * (n - 1) + 2 * (i - 1) + 1, c + 1] = 2
            A[2 * (n - 1) + 2 * (i - 1) + 1, c + 4] = -6 * x[i]
            A[2 * (n - 1) + 2 * (i - 1) + 1, c + 5] = -2
            b[2 * (n - 1) + 2 * (i - 1) + 1] = 0
            c += 4

        A[-2, 0] = 6 * x[0]
        A[-2, 1] = 2
        b[-2] = 0

        A[-1, -4] = 6 * x[-1]
        A[-1, -3] = 2
        b[-1] = 0

    coef = np.linalg.solve(A, b)
    for i in range(n - 1):
        if degree == 1:
            coefficients.append((i, coef[2 * i], coef[2 * i + 1]))
            polynomials.append([coef[2 * i], coef[2 * i + 1]])
        elif degree == 2:
            coefficients.append((i, coef[3 * i], coef[3 * i + 1], coef[3 * i + 2]))
            polynomials.append([coef[3 * i], coef[3 * i + 1], coef[3 * i + 2]])
        elif degree == 3:
            coefficients.append((i, coef[4 * i], coef[4 * i + 1], coef[4 * i + 2], coef[4 * i + 3]))
            polynomials.append([coef[4 * i], coef[4 * i + 1], coef[4 * i + 2], coef[4 * i + 3]])
    return coefficients, polynomials