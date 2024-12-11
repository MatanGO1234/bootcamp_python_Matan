"""Module contain only the Calculator Class."""

from package_name.logger import logger


class Calculator:
    """Calculator is a static Class that perform the following operations:

    - Addition
    - Divition
    """

    @staticmethod
    def add(a: float, b: float) -> float:
        """The sum of two numbers.

        Parameters
        ----------
        a : float - the first number `a`
        b : float - the second number `b`

        Returns
        -------
        float
        The sum of `a` and `b`

        """
        return a + b

    @staticmethod
    def divide(a: float, b: float) -> float:
        """The divition of two numbers.

        Parameters
        ----------
        a : float - the dividend `a`
        b : float - the divisor `b`

        Returns
        -------
        float
        `a` / `b`

        Raises:
             ZeroDivisionError: if b value is 0
        """
        try:
            return a / b
        except ZeroDivisionError:
            logger.error("Cannot divide by zero")
            raise
