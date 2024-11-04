# Contributing to MLCookieCutter

Thank you for your interest in contributing to MLCookieCutter! We welcome contributions that help improve the quality, usability, and functionality of this project. Whether you're fixing bugs, adding new templates, or improving documentation, your work is highly appreciated.

## How to Contribute

### 1. Fork the Repository
- Go to the [MLCookieCutter GitHub page](https://github.com/j-raghavan/MLCookieCutter).
- Click on the "Fork" button in the upper right corner to create your own copy of the repository.

### 2. Clone the Repository
Clone your fork to your local machine:


    git clone https://github.com/j-raghavan MLCookieCutter.git
    cd MLCookieCutter


3. Create a Branch
Create a branch to work on, ideally with a descriptive name:

    ```bash
    git checkout -b feature/new-model-template
    ```

4. Install Dependencies
We use Poetry for package management, so install all dependencies (including development dependencies) with:

    ```bash
    poetry install
    ```

5. Make Your Changes
Add new model templates, fix bugs, improve existing templates, or update documentation.
Follow existing code style guidelines and ensure the code is well-documented.

6. Format and Lint Your Code
We use ruff to ensure code quality and style consistency. Run ruff to check your code for any linting issues:

```bash
poetry run ruff check .
```

If ruff identifies any issues, you can have it automatically fix many of them with:

```bash
poetry run ruff check . --fix
```

7. Run Tests
Ensure that all tests pass before submitting your changes:

```bash
poetry run pytest
```

8. Submit a Pull Request (PR)
Push your changes to your fork and create a pull request:

```bash
git push origin feature/new-model-template
```
Then, go to the original repository and click on "New Pull Request."

## Code Style
- Follow PEP 8 guidelines for Python code.
- Use ruff to lint and format your code as described above.
- Write descriptive commit messages.
- Ensure each new model template is placed in the templates/ folder and follows a similar structure to existing templates.

## Additional Resources
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [pytest Documentation](https://docs.pytest.org/en/stable/contents.html)


Thank you for helping make MLCookieCutter better!