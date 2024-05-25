from django.contrib import admin
from django.urls import path
from . import views 

urlpatterns = [
    path('', views.Home, name="home"),
    path('methods/', views.Methods, name="methods"),
    path('about/', views.about_view, name='about'),
    path('help/', views.help, name='help'),
    path('bisection_method/', views.bisection_method, name="bisection_method"),
    path('false_position/', views.false_position_method, name="false_position_method"),
    path('fixed_point/', views.fixed_point_method, name="fixed_point_method"),
    path('newton/', views.newton_method, name="newton_method"),
    path('secant/', views.secant_method, name="secant_method"),
    path('multiple_roots/', views.multiple_roots_method, name="multiple_roots_method"),
    path('gauss_seidel/', views.gauss_method, name="gauss_method"),
    path('sor/', views.sor_method, name="sor_method"),
    path('jacobi/', views.jacobi_method, name="jacobi_method"),
    path('vandermonde/', views.vandermonde_method, name="vandermonde_method"),
    path('newton_interpolation/', views.newton_interpolation_method, name="newton_interpolation_method"),
    path('lagrange/', views.lagrange_interpolation_method, name="lagrange_interpolation_method"),


]
