"""Experimentation Area for Package Modules"""

from package_name.logger import logger
from package_name.tools.calculator import Calculator

if __name__ == "__main__":
    logger.info(f"1+3={Calculator.add(1,3)}")
