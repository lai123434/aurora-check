# Contributing to Aurora Check

First off, thank you for considering contributing to Aurora Check! It's people like you that make Aurora Check such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](https://github.com/lai123434/aurora-check/issues) to see if someone else in the community has already created a ticket. If not, go ahead and [make one](https://github.com/lai123434/aurora-check/issues/new)!

## Fork & create a branch

If this is something you think you can fix, then [fork Aurora Check](https://github.com/lai123434/aurora-check/fork) and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```bash
git checkout -b 325-add-japanese-localization
```

## Get the code running

```bash
# Clone your fork
git clone https://github.com/your-username/aurora-check.git
cd aurora-check

# Install dependencies
pip install -r requirements.txt
```

## Make your changes

Do your best to follow these guidelines when making your changes:

- Keep commits small and focused
- Write meaningful commit messages
- Add tests for new features (if applicable)
- Update documentation as needed

## Test your changes

```bash
# Run the detector on the sample essay
python aurora_check.py examples/sample_essay.txt

# Make sure it doesn't crash
# And the output looks reasonable
```

## Did you find a bug?

- **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/lai123434/aurora-check/issues).

- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/lai123434/aurora-check/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample** or an **executable test case** demonstrating the expected behavior that is not occurring.

## Did you write a patch that fixes a bug?

- Open a new GitHub pull request with the patch.

- Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.

## Do you intend to add a new feature or change an existing one?

- Suggest your change in the [Issues](https://github.com/lai123434/aurora-check/issues) and start writing code.

- Do not open a pull request until you have collected positive feedback about the change.

## Code of Conduct

Be nice. Be respectful. We're all here to build something great together.

Thanks! ❤️
Aurora Check Team
