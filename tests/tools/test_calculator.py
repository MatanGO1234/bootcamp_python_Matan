"""Test Calculator Class."""

import pytest

from package_name.tools.calculator import Calculator


@pytest.fixture
def calculator() -> type[Calculator]:
    """Fixture to create a Calculator instance."""
    return Calculator


# Positive test cases
def test_add_positive(calculator: Calculator) -> None:
    """Test add method with positive numbers."""
    expected_value = 15
    assert calculator.add(10, 5) == expected_value


def test_divide_positive(calculator: Calculator) -> None:
    """Test divide method with valid input values."""
    expected_value = 5.0
    assert calculator.divide(10, 2) == expected_value


# Negative test cases
def test_add_negative(calculator: Calculator) -> None:
    """Test add method with negative numbers."""
    expected_value = -10
    assert calculator.add(-3, -7) == expected_value


def test_divide_by_zero(calculator: Calculator) -> None:
    """Test divide method raises ValueError for division by zero."""
    with pytest.raises(ZeroDivisionError):
        calculator.divide(10, 0)


# Boundary test cases
def test_add_boundary_zero(calculator: Calculator) -> None:
    """Test add method when one value is zero."""
    expected_value = 5
    assert calculator.add(0, 5) == expected_value


def test_divide_boundary_zero(calculator: Calculator) -> None:
    """Test divide method when numerator is zero."""
    expected_value = 0.0
    assert calculator.divide(0, 5) == expected_value


def test_divide_boundary_near_zero(calculator: Calculator) -> None:
    """Test divide method with very small non-zero denominator."""
    expected_value = 100000000.0
    assert pytest.approx(calculator.divide(10, 1e-7), rel=1e-9) == expected_value


# Edge cases
def test_add_large_numbers(calculator: Calculator) -> None:
    """Test add method with large floating-point numbers."""
    expected_value = 2e18
    assert calculator.add(1e18, 1e18) == expected_value


def test_divide_large_numbers(calculator: Calculator) -> None:
    """Test divide method with large floating-point numbers."""
    expected_value = 1e12
    assert calculator.divide(1e18, 1e6) == expected_value


def test_add_float_precision(calculator: Calculator) -> None:
    """Test add method with floating-point precision."""
    expected_value = 0.3
    assert pytest.approx(calculator.add(0.1, 0.2), rel=1e-9) == expected_value


def test_divide_float_precision(calculator: Calculator) -> None:
    """Test divide method with floating-point precision."""
    expected_value = 0.3333333333333333
    assert pytest.approx(calculator.divide(1.0, 3.0), rel=1e-9) == expected_value
