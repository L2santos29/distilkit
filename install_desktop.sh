#!/bin/bash
# Install DistilKit desktop entry for your user account.
# Run this once to add DistilKit to your application menu.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_FILE="$SCRIPT_DIR/distilkit.desktop"
INSTALL_DIR="$HOME/.local/share/applications"

# Update paths in the .desktop file to match the current location
sed -i "s|Exec=.*|Exec=$SCRIPT_DIR/run_gui.sh|" "$DESKTOP_FILE"
sed -i "s|Icon=.*|Icon=$SCRIPT_DIR/assets/icon.svg|" "$DESKTOP_FILE"

mkdir -p "$INSTALL_DIR"
cp "$DESKTOP_FILE" "$INSTALL_DIR/distilkit.desktop"

echo "✅ DistilKit installed in your application menu!"
echo "   Look for 'DistilKit' in your launcher."
echo ""
echo "   To uninstall: rm $INSTALL_DIR/distilkit.desktop"
