# Deployment

Bump version:
- `hatch version fix`   - "0.1.0" -> "0.1.1"
- `hatch version minor` - "0.1.0" -> "0.2.0"
- `hatch version major` - "0.1.0" -> "1.0.0"

Build:
- `python -m build -s`
- `python -m build -w`
- `python -m build -w -C--global-option="--plat-name win-amd64"` - force platform
- `python -m build` - both

Publish:
- `twine upload dist/*`
- `hatch publish`
