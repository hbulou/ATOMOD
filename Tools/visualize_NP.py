"""
visualize_NP.py
───────────────
Visualiseur autonome de nanoparticules au format XYZ.
Génère une page HTML interactive (3Dmol.js) et l'ouvre dans le navigateur.

Utilisation :
    python visualize_NP.py NP.xyz
    python visualize_NP.py NP.xyz NP2.xyz NP3.xyz   ← plusieurs fichiers
    python visualize_NP.py data/xyz/                 ← dossier entier

Dépendances Python : aucune (stdlib uniquement)
Dépendance réseau  : 3Dmol.js chargé depuis CDN (ou bundle local si offline)
"""

import sys
import os
import argparse
import tempfile
import webbrowser
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# COULEURS PAR ÉLÉMENT (schéma Jmol standard)
# ═══════════════════════════════════════════════════════════════════════════

JMOL_COLORS = {
    # Éléments courants nanoparticules HEA
    'Rh': '#0A89FF', 'Ir': '#175487', 'Pt': '#D0D0E0',
    'Pd': '#006985', 'Au': '#FFD123', 'Ag': '#C0C0C0',
    'Co': '#F090A0', 'Ni': '#50D050', 'Fe': '#E06633',
    'Cu': '#C88033', 'Ru': '#248F8F', 'Os': '#266696',
    # Autres
    'H': '#FFFFFF',  'C': '#909090', 'N': '#3050F8',
    'O': '#FF0D0D',  'S': '#FFFF30', 'P': '#FF8000',
    'Cl': '#1FF01F', 'F': '#90E050',
}

RAYON_COVALENT = {
    'Rh': 1.34, 'Ir': 1.41, 'Pt': 1.36, 'Pd': 1.31,
    'Au': 1.36, 'Ag': 1.45, 'Co': 1.26, 'Ni': 1.24,
    'Fe': 1.26, 'Cu': 1.28, 'Ru': 1.36, 'Os': 1.44,
    'H':  0.31, 'C':  0.76, 'N':  0.71, 'O':  0.66,
}

def couleur(elt):
    return JMOL_COLORS.get(elt, '#FF1493')

def rayon(elt):
    return RAYON_COVALENT.get(elt, 1.5)


# ═══════════════════════════════════════════════════════════════════════════
# PARSING XYZ
# ═══════════════════════════════════════════════════════════════════════════

def lire_xyz(chemin):
    """
    Lit un fichier XYZ et retourne (n_atomes, commentaire, liste_atomes).
    Format standard :
        N
        commentaire
        Elt  x  y  z
        ...
    """
    lignes = Path(chemin).read_text(encoding='utf-8').strip().splitlines()
    try:
        n = int(lignes[0].strip())
    except ValueError:
        raise ValueError(f"Format XYZ invalide dans {chemin} : ligne 1 doit être le nombre d'atomes")

    commentaire = lignes[1].strip() if len(lignes) > 1 else ''
    atomes = []

    for i, ligne in enumerate(lignes[2:2 + n], start=2):
        parties = ligne.split()
        if len(parties) < 4:
            continue
        elt = parties[0]
        x, y, z = float(parties[1]), float(parties[2]), float(parties[3])
        atomes.append({'elt': elt, 'x': x, 'y': y, 'z': z})

    return n, commentaire, atomes


def statistiques(atomes):
    """Retourne un dict espèce → nombre d'atomes."""
    stats = {}
    for a in atomes:
        stats[a['elt']] = stats.get(a['elt'], 0) + 1
    return dict(sorted(stats.items()))


# ═══════════════════════════════════════════════════════════════════════════
# GÉNÉRATION HTML
# ═══════════════════════════════════════════════════════════════════════════

def generer_html(fichiers_xyz):
    """
    Génère une page HTML complète avec :
    - Un onglet par fichier XYZ
    - Visionneuse 3Dmol.js interactive
    - Panneau d'information (composition, nb atomes)
    - Contrôles de style (sphère / bâton / VdW)
    """

    # ── Préparer les données de chaque fichier ──
    structures = []
    for chemin in fichiers_xyz:
        n, commentaire, atomes = lire_xyz(chemin)
        stats = statistiques(atomes)
        nom = Path(chemin).name
        xyz_raw = Path(chemin).read_text(encoding='utf-8')
        structures.append({
            'nom': nom,
            'chemin': str(chemin),
            'n': n,
            'commentaire': commentaire,
            'atomes': atomes,
            'stats': stats,
            'xyz_raw': xyz_raw,
        })

    # ── Construire les données JSON pour JavaScript ──
    import json

    structs_js = json.dumps([
        {
            'nom':        s['nom'],
            'chemin':     s['chemin'],
            'n':          s['n'],
            'commentaire':s['commentaire'],
            'stats':      s['stats'],
            'xyz':        s['xyz_raw'],
        }
        for s in structures
    ], ensure_ascii=False)

    colors_js  = json.dumps(JMOL_COLORS)
    rayons_js  = json.dumps(RAYON_COVALENT)

    # ── HTML ──
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ATOMOD — Visualiseur NP</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }}

  /* ── Barre du haut ── */
  header {{
    background: #1a1d2e;
    border-bottom: 1px solid #2d3148;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }}
  header h1 {{
    font-size: 15px;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: 0.05em;
  }}
  header span {{
    font-size: 12px;
    color: #64748b;
  }}

  /* ── Onglets ── */
  #tabs {{
    display: flex;
    gap: 4px;
    padding: 8px 16px 0;
    background: #1a1d2e;
    border-bottom: 1px solid #2d3148;
    flex-shrink: 0;
    flex-wrap: wrap;
  }}
  .tab {{
    padding: 6px 14px;
    border-radius: 6px 6px 0 0;
    font-size: 12px;
    cursor: pointer;
    background: #252840;
    color: #94a3b8;
    border: 1px solid #2d3148;
    border-bottom: none;
    transition: all 0.15s;
    white-space: nowrap;
  }}
  .tab:hover  {{ background: #2d3058; color: #e2e8f0; }}
  .tab.active {{ background: #0f1117; color: #a78bfa; border-color: #4c4f8a; }}

  /* ── Contenu principal ── */
  #main {{
    display: flex;
    flex: 1;
    overflow: hidden;
  }}

  /* ── Panneau info (gauche) ── */
  #info-panel {{
    width: 240px;
    flex-shrink: 0;
    background: #1a1d2e;
    border-right: 1px solid #2d3148;
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}

  .section-title {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4c4f8a;
    margin-bottom: 8px;
  }}

  #file-name {{
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    word-break: break-all;
  }}
  #file-comment {{
    font-size: 11px;
    color: #64748b;
    margin-top: 4px;
    font-style: italic;
  }}
  #n-atoms {{
    font-size: 22px;
    font-weight: 700;
    color: #a78bfa;
  }}
  #n-atoms-label {{
    font-size: 11px;
    color: #64748b;
  }}

  /* Composition */
  #composition-list {{
    display: flex;
    flex-direction: column;
    gap: 5px;
  }}
  .elt-row {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .elt-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    border: 1px solid rgba(255,255,255,0.15);
  }}
  .elt-name {{
    font-size: 12px;
    font-weight: 600;
    width: 28px;
  }}
  .elt-bar-wrap {{
    flex: 1;
    background: #252840;
    border-radius: 3px;
    height: 6px;
  }}
  .elt-bar {{
    height: 6px;
    border-radius: 3px;
    transition: width 0.4s;
  }}
  .elt-count {{
    font-size: 11px;
    color: #94a3b8;
    width: 38px;
    text-align: right;
  }}

  /* Contrôles de style */
  .controls {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .ctrl-label {{
    font-size: 11px;
    color: #94a3b8;
  }}
  .btn-group {{
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }}
  .btn {{
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 11px;
    cursor: pointer;
    background: #252840;
    color: #94a3b8;
    border: 1px solid #2d3148;
    transition: all 0.15s;
  }}
  .btn:hover  {{ background: #2d3058; color: #e2e8f0; }}
  .btn.active {{ background: #4c4f8a; color: #e2e8f0; border-color: #6366f1; }}

  /* ── Visionneuse 3D ── */
  #viewer-wrap {{
    flex: 1;
    position: relative;
    background: #050508;
  }}
  #viewer {{
    width: 100%;
    height: 100%;
  }}

  /* Légende flottante */
  #legend {{
    position: absolute;
    bottom: 12px;
    right: 12px;
    background: rgba(26,29,46,0.85);
    backdrop-filter: blur(6px);
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    max-height: 220px;
    overflow-y: auto;
  }}
  .legend-row {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.2);
  }}
</style>
</head>
<body>

<header>
  <h1>ATOMOD — Visualiseur Nanoparticules</h1>
  <span id="header-info"></span>
</header>

<div id="tabs"></div>

<div id="main">
  <!-- Panneau info -->
  <div id="info-panel">
    <div>
      <div class="section-title">Fichier</div>
      <div id="file-name">—</div>
      <div id="file-comment"></div>
    </div>
    <div>
      <div class="section-title">Atomes</div>
      <div id="n-atoms">—</div>
      <div id="n-atoms-label">atomes</div>
    </div>
    <div>
      <div class="section-title">Composition</div>
      <div id="composition-list"></div>
    </div>
    <div>
      <div class="section-title">Style</div>
      <div class="controls">
        <div class="ctrl-label">Représentation</div>
        <div class="btn-group">
          <button class="btn active" id="btn-sphere" onclick="setStyle('sphere')">Sphères</button>
          <button class="btn"        id="btn-stick"  onclick="setStyle('stick')">Bâtons</button>
          <button class="btn"        id="btn-vdw"    onclick="setStyle('vdw')">VdW</button>
        </div>
        <div class="ctrl-label">Fond</div>
        <div class="btn-group">
          <button class="btn active" id="btn-dark"  onclick="setBg('black')">Sombre</button>
          <button class="btn"        id="btn-light" onclick="setBg('white')">Clair</button>
          <button class="btn"        id="btn-grey"  onclick="setBg('#1a1a2e')">Bleu</button>
        </div>
        <div class="ctrl-label">Actions</div>
        <div class="btn-group">
          <button class="btn" onclick="viewer.zoomTo()">Recentrer</button>
          <button class="btn" onclick="viewer.spin(!spinning); spinning=!spinning; this.classList.toggle('active')">Rotation</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Visionneuse 3D -->
  <div id="viewer-wrap">
    <div id="viewer"></div>
    <div id="legend"></div>
  </div>
</div>

<script>
// ── Données injectées depuis Python ──
const STRUCTURES = {structs_js};
const JMOL_COLORS = {colors_js};
const RAYONS     = {rayons_js};

// ── État global ──
let viewer       = null;
let currentIdx   = 0;
let currentStyle = 'sphere';
let spinning     = false;

// ── Initialisation ──
document.addEventListener('DOMContentLoaded', () => {{
  creerOnglets();
  initViewer();
  chargerStructure(0);

  document.getElementById('header-info').textContent =
    `${{STRUCTURES.length}} structure${{STRUCTURES.length > 1 ? 's' : ''}}`;
}});

function creerOnglets() {{
  const tabsEl = document.getElementById('tabs');
  STRUCTURES.forEach((s, i) => {{
    const t = document.createElement('div');
    t.className = 'tab' + (i === 0 ? ' active' : '');
    t.textContent = s.nom;
    t.title = s.chemin;
    t.onclick = () => chargerStructure(i);
    tabsEl.appendChild(t);
  }});
}}

function initViewer() {{
  const el = document.getElementById('viewer');
  viewer = $3Dmol.createViewer(el, {{
    backgroundColor: 'black',
    antialias: true,
  }});
}}

function chargerStructure(idx) {{
  currentIdx = idx;
  const s = STRUCTURES[idx];

  // Activer l'onglet
  document.querySelectorAll('.tab').forEach((t, i) => {{
    t.classList.toggle('active', i === idx);
  }});

  // Mettre à jour le panneau info
  document.getElementById('file-name').textContent = s.nom;
  document.getElementById('file-comment').textContent = s.commentaire || '';
  document.getElementById('n-atoms').textContent = s.n.toLocaleString();

  // Composition
  afficherComposition(s.stats, s.n);

  // Visionneuse
  viewer.clear();
  viewer.addModel(s.xyz, 'xyz');
  appliquerStyle(currentStyle);
  viewer.zoomTo();
  viewer.render();

  // Légende
  afficherLegende(s.stats);
}}

function couleurElt(elt) {{
  return JMOL_COLORS[elt] || '#FF1493';
}}
function rayonElt(elt) {{
  return (RAYONS[elt] || 1.5) * 0.6;
}}

function appliquerStyle(style) {{
  viewer.setStyle({{}}, {{}});  // reset

  if (style === 'sphere') {{
    viewer.setStyle({{}}, {{
      sphere: {{
        colorscheme: {{ prop: 'elem', map: JMOL_COLORS }},
        radius: 0.6,
      }}
    }});
  }} else if (style === 'stick') {{
    viewer.setStyle({{}}, {{
      stick: {{
        colorscheme: {{ prop: 'elem', map: JMOL_COLORS }},
        radius: 0.15,
      }},
      sphere: {{
        colorscheme: {{ prop: 'elem', map: JMOL_COLORS }},
        radius: 0.25,
      }}
    }});
  }} else if (style === 'vdw') {{
    viewer.setStyle({{}}, {{
      sphere: {{
        colorscheme: {{ prop: 'elem', map: JMOL_COLORS }},
        radius: 1.0,
      }}
    }});
  }}
  viewer.render();
}}

function setStyle(style) {{
  currentStyle = style;
  ['sphere','stick','vdw'].forEach(s => {{
    document.getElementById('btn-' + s).classList.toggle('active', s === style);
  }});
  appliquerStyle(style);
}}

function setBg(color) {{
  viewer.setBackgroundColor(color);
  viewer.render();
  ['dark','light','grey'].forEach(s =>
    document.getElementById('btn-' + s).classList.remove('active')
  );
  const map = {{ 'black':'dark', 'white':'light', '#1a1a2e':'grey' }};
  if (map[color]) document.getElementById('btn-' + map[color]).classList.add('active');
}}

function afficherComposition(stats, total) {{
  const el = document.getElementById('composition-list');
  el.innerHTML = '';
  Object.entries(stats).sort((a,b) => b[1]-a[1]).forEach(([elt, n]) => {{
    const pct = (n / total * 100).toFixed(1);
    const col = couleurElt(elt);
    el.innerHTML += `
      <div class="elt-row">
        <div class="elt-dot" style="background:${{col}}"></div>
        <div class="elt-name">${{elt}}</div>
        <div class="elt-bar-wrap">
          <div class="elt-bar" style="width:${{pct}}%;background:${{col}}"></div>
        </div>
        <div class="elt-count">${{n}} <span style="color:#4c4f8a;font-size:10px">(${{pct}}%)</span></div>
      </div>`;
  }});
}}

function afficherLegende(stats) {{
  const el = document.getElementById('legend');
  el.innerHTML = '';
  Object.entries(stats).sort((a,b) => b[1]-a[1]).forEach(([elt, n]) => {{
    el.innerHTML += `
      <div class="legend-row">
        <div class="legend-dot" style="background:${{couleurElt(elt)}}"></div>
        <span style="font-weight:600;width:28px">${{elt}}</span>
        <span style="color:#64748b">${{n}} at.</span>
      </div>`;
  }});
}}
</script>
</body>
</html>"""
    return html


# ═══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════

def collecter_fichiers(args_paths):
    """Collecte tous les fichiers XYZ depuis les chemins fournis."""
    fichiers = []
    for p in args_paths:
        path = Path(p)
        if path.is_dir():
            fichiers.extend(sorted(path.glob('*.xyz')))
        elif path.is_file() and path.suffix.lower() == '.xyz':
            fichiers.append(path)
        else:
            print(f"⚠️  Ignoré (pas un .xyz ou dossier) : {p}")
    return fichiers


def main():
    parser = argparse.ArgumentParser(
        description='Visualiseur de nanoparticules XYZ — génère une page HTML interactive',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python visualize_NP.py NP.xyz
  python visualize_NP.py NP_0001.xyz NP_0002.xyz
  python visualize_NP.py data/xyz/
  python visualize_NP.py data/xyz/ --output ma_visu.html --no-open
        """
    )
    parser.add_argument('chemins', nargs='+',
                        help='Fichier(s) .xyz ou dossier contenant des .xyz')
    parser.add_argument('--output', '-o', default=None,
                        help='Nom du fichier HTML de sortie (défaut: temp)')
    parser.add_argument('--no-open', action='store_true',
                        help='Ne pas ouvrir le navigateur automatiquement')

    args = parser.parse_args()

    # Collecter les fichiers
    fichiers = collecter_fichiers(args.chemins)

    if not fichiers:
        print("❌  Aucun fichier .xyz trouvé.")
        sys.exit(1)

    print(f"📂  {len(fichiers)} fichier(s) XYZ trouvé(s) :")
    for f in fichiers:
        n, _, atomes = lire_xyz(f)
        stats = statistiques(atomes)
        comp = ', '.join(f'{e}:{c}' for e, c in stats.items())
        print(f"    • {f.name}  ({n} atomes  —  {comp})")

    # Générer le HTML
    print("\n🔧  Génération de la page HTML...")
    html = generer_html(fichiers)

    # Écrire le fichier
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(html, encoding='utf-8')
        print(f"✅  Page HTML sauvegardée : {out_path.resolve()}")
    else:
        # Fichier temporaire (ouvert dans le navigateur)
        tmp = tempfile.NamedTemporaryFile(
            suffix='.html', delete=False,
            mode='w', encoding='utf-8', prefix='atomod_NP_'
        )
        tmp.write(html)
        tmp.close()
        out_path = Path(tmp.name)
        print(f"📄  Page HTML temporaire : {out_path}")

    # Ouvrir le navigateur
    if not args.no_open:
        print("🌐  Ouverture dans le navigateur...")
        webbrowser.open(out_path.as_uri())

    print("\n💡  Contrôles 3D :")
    print("    • Clic gauche + glisser  : rotation")
    print("    • Clic droit + glisser   : translation")
    print("    • Molette                : zoom")
    print("    • Boutons panneau gauche : style / fond / rotation auto")


if __name__ == '__main__':
    main()
