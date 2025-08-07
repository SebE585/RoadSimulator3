#!/bin/bash

grep -r --include="*.py" -E "^def .*search.*\(" . | while IFS=: read -r filepath lineno line; do
  # Extraire le nom de la fonction (entre def et parenthèse)
  funcname=$(echo "$line" | sed -E 's/^def ([^(]+).*/\1/')

  # Retirer './' et '.py', remplacer '/' par '.'
  modpath=${filepath#./}
  modpath=${modpath%.py}
  modpath=${modpath//\//.}

  echo "$filepath => from $modpath import $funcname"
done