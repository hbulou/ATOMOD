import subprocess
import sys

def run_git_command(command):
    """Exécute une commande shell et gère les erreurs éventuelles."""
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'exécution de : {' '.join(command)}")
        print(e.stderr)
        sys.exit(1)

def reset_repository():
    print("🔄 Récupération des dernières informations depuis GitHub...")
    run_git_command(['git', 'fetch', 'origin'])

    print("⚠️ Réinitialisation forcée d'ATOMOD (les modifications locales seront écrasées)...")
    run_git_command(['git', 'reset', '--hard', 'origin/main'])
    
    print("✅ ATOMOD a été réinitialisé avec succès et synchronisé sur la version officielle !")

if __name__ == "__main__":
    reset_repository()
