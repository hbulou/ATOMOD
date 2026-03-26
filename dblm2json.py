import re
import json
from typing import Dict, List, Any
from pathlib import Path


class DBMLToMongoConverter:
    """
    Convertit un fichier DBML en collections MongoDB (format JSON).
    
    Le DBML est un format de définition de schéma de base de données.
    Ce script extrait les tables et les convertit en schémas MongoDB.
    """
    
    def __init__(self, dbml_file: str):
        self.dbml_file = dbml_file
        self.collections = {}
        self.relationships = []
        
    def parse_dbml(self) -> Dict[str, Any]:
        """Parse le fichier DBML et extrait les tables."""
        with open(self.dbml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extraction des tables
        table_pattern = r'Table\s+(\w+)\s*\{([^}]+)\}'
        tables = re.findall(table_pattern, content, re.DOTALL)
        
        for table_name, table_body in tables:
            self.collections[table_name] = self._parse_table(table_name, table_body)
        
        # Extraction des relations (Ref)
        ref_pattern = r'Ref:\s*(\w+)\.(\w+)\s*([<>-]+)\s*(\w+)\.(\w+)'
        refs = re.findall(ref_pattern, content)
        
        for ref in refs:
            self.relationships.append({
                'from_table': ref[0],
                'from_field': ref[1],
                'type': ref[2],
                'to_table': ref[3],
                'to_field': ref[4]
            })
        
        return self.collections
    
    def _parse_table(self, table_name: str, table_body: str) -> Dict[str, Any]:
        """Parse le corps d'une table DBML."""
        schema = {
            '_collection': table_name,
            'fields': {},
            'indexes': [],
            'constraints': []
        }
        
        # Extraction des champs
        field_pattern = r'(\w+)\s+(\w+)(?:\(([^)]+)\))?(?:\s+\[([^\]]+)\])?'
        fields = re.findall(field_pattern, table_body)
        
        for field_name, field_type, type_params, constraints in fields:
            if field_name.lower() in ['note', 'indexes']:  # Ignorer les mots-clés
                continue
                
            field_def = {
                'type': self._convert_type(field_type, type_params),
                'required': False,
                'unique': False,
                'default': None
            }
            
            # Parser les contraintes
            if constraints:
                field_def.update(self._parse_constraints(constraints))
            
            schema['fields'][field_name] = field_def
        
        # Extraction des indexes
        index_pattern = r'indexes\s*\{([^}]+)\}'
        index_match = re.search(index_pattern, table_body, re.DOTALL)
        if index_match:
            schema['indexes'] = self._parse_indexes(index_match.group(1))
        
        return schema
    
    def _convert_type(self, dbml_type: str, params: str = None) -> str:
        """Convertit un type DBML en type MongoDB/JSON Schema."""
        type_mapping = {
            'int': 'integer',
            'integer': 'integer',
            'bigint': 'long',
            'smallint': 'integer',
            'varchar': 'string',
            'text': 'string',
            'char': 'string',
            'boolean': 'boolean',
            'bool': 'boolean',
            'date': 'date',
            'datetime': 'date',
            'timestamp': 'date',
            'float': 'double',
            'double': 'double',
            'decimal': 'decimal',
            'json': 'object',
            'jsonb': 'object',
            'array': 'array'
        }
        
        base_type = type_mapping.get(dbml_type.lower(), 'string')
        
        # Gestion des paramètres (ex: varchar(255))
        if params and base_type == 'string':
            return f"string (max: {params})"
        
        return base_type
    
    def _parse_constraints(self, constraint_str: str) -> Dict[str, Any]:
        """Parse les contraintes d'un champ (pk, unique, not null, default, etc.)."""
        constraints = {}
        
        if 'pk' in constraint_str.lower() or 'primary key' in constraint_str.lower():
            constraints['primary_key'] = True
            constraints['required'] = True
        
        if 'not null' in constraint_str.lower():
            constraints['required'] = True
        
        if 'unique' in constraint_str.lower():
            constraints['unique'] = True
        
        if 'increment' in constraint_str.lower():
            constraints['auto_increment'] = True
        
        # Extraction de la valeur par défaut
        default_pattern = r'default:\s*["\']?([^,\]]+)["\']?'
        default_match = re.search(default_pattern, constraint_str)
        if default_match:
            constraints['default'] = default_match.group(1).strip()
        
        # Extraction de la note
        note_pattern = r'note:\s*["\']([^"\']+)["\']'
        note_match = re.search(note_pattern, constraint_str)
        if note_match:
            constraints['note'] = note_match.group(1)
        
        return constraints
    
    def _parse_indexes(self, index_body: str) -> List[Dict[str, Any]]:
        """Parse les définitions d'index."""
        indexes = []
        
        # Pattern pour les indexes simples ou composites
        # Ex: (user_id, post_id) [name: 'user_post_idx']
        index_pattern = r'\(([^)]+)\)(?:\s*\[([^\]]+)\])?'
        index_matches = re.findall(index_pattern, index_body)
        
        for fields_str, options_str in index_matches:
            fields = [f.strip() for f in fields_str.split(',')]
            
            index_def = {
                'fields': fields,
                'unique': False,
                'name': None
            }
            
            if options_str:
                if 'unique' in options_str.lower():
                    index_def['unique'] = True
                
                name_match = re.search(r'name:\s*["\']([^"\']+)["\']', options_str)
                if name_match:
                    index_def['name'] = name_match.group(1)
                else:
                    index_def['name'] = '_'.join(fields) + '_idx'
            else:
                index_def['name'] = '_'.join(fields) + '_idx'
            
            indexes.append(index_def)
        
        return indexes
    
    def add_embedded_documents(self):
        """
        Transforme les relations 1-N en documents embarqués (approche MongoDB).
        
        Par exemple, si User -> Posts (1-N), on peut embarquer les posts dans User.
        """
        for rel in self.relationships:
            if rel['type'] in ['<', '>']:  # Relations 1-N
                parent = rel['from_table'] if rel['type'] == '<' else rel['to_table']
                child = rel['to_table'] if rel['type'] == '<' else rel['from_table']
                
                # Ajouter un champ array dans le parent
                if parent in self.collections:
                    self.collections[parent]['fields'][f'{child.lower()}_embedded'] = {
                        'type': 'array',
                        'items': f'Reference to {child}',
                        'note': f'Embedded {child} documents (1-N relationship)'
                    }
    
    def generate_mongodb_schema(self) -> Dict[str, Any]:
        """Génère le schéma MongoDB complet avec validation."""
        mongodb_schemas = {}
        
        for collection_name, collection_def in self.collections.items():
            schema = {
                'collectionName': collection_name,
                'validator': {
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': [],
                        'properties': {}
                    }
                },
                'indexes': []
            }
            
            # Construire le JSON Schema
            for field_name, field_def in collection_def['fields'].items():
                prop = {'bsonType': self._mongo_type(field_def['type'])}
                
                if field_def.get('default'):
                    prop['default'] = field_def['default']
                
                if field_def.get('note'):
                    prop['description'] = field_def['note']
                
                schema['validator']['$jsonSchema']['properties'][field_name] = prop
                
                if field_def.get('required'):
                    schema['validator']['$jsonSchema']['required'].append(field_name)
            
            # Ajouter les indexes
            for index in collection_def.get('indexes', []):
                index_spec = {
                    'key': {field: 1 for field in index['fields']},
                    'name': index['name'],
                    'unique': index['unique']
                }
                schema['indexes'].append(index_spec)
            
            mongodb_schemas[collection_name] = schema
        
        return mongodb_schemas
    
    def _mongo_type(self, json_type: str) -> str:
        """Convertit un type JSON en type BSON MongoDB."""
        type_map = {
            'integer': 'int',
            'long': 'long',
            'double': 'double',
            'decimal': 'decimal',
            'string': 'string',
            'boolean': 'bool',
            'date': 'date',
            'object': 'object',
            'array': 'array'
        }
        
        # Gérer les types avec paramètres (ex: "string (max: 255)")
        base_type = json_type.split()[0] if ' ' in json_type else json_type
        
        return type_map.get(base_type, 'string')
    
    def export_to_json(self, output_file: str, include_relationships: bool = True):
        """Exporte le schéma MongoDB en fichier JSON."""
        schemas = self.generate_mongodb_schema()
        
        output = {
            'version': '1.0',
            'database': Path(self.dbml_file).stem,
            'collections': schemas
        }
        
        if include_relationships:
            output['relationships'] = self.relationships
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Schéma MongoDB exporté vers: {output_file}")
        print(f"📊 {len(schemas)} collections générées")
    
    def export_to_separate_files(self, output_dir: str):
        """Exporte chaque collection dans un fichier JSON séparé."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        schemas = self.generate_mongodb_schema()
        
        for collection_name, schema in schemas.items():
            file_path = output_path / f"{collection_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
            print(f"✅ {collection_name} -> {file_path}")
        
        # Export des relations
        if self.relationships:
            rel_file = output_path / "relationships.json"
            with open(rel_file, 'w', encoding='utf-8') as f:
                json.dump({'relationships': self.relationships}, f, indent=2, ensure_ascii=False)
            print(f"✅ Relations -> {rel_file}")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

def main():
    """Exemple d'utilisation du convertisseur."""
    
    # 1. Créer un fichier DBML d'exemple
    example_dbml = """
// Use DBML to define your database structure
// Docs: https://dbml.dbdiagram.io/docs


Table samples {
  synthesis_id integer	
  name varchar [primary key]
}


Table characterizations {
  id integer [primary key]
  operator text [note: "Name of the operator who performed the caracterisation"]
  method text [note: "Name of the method used to characterized the sample"]
  created_at timestamp
  sample varchar 
}





Table synthesis {
  id integer [primary key]
  method text [note: "Name of the method used to manufacture the sample"]
  manufacturer text [note: "Name of the operator who manufactured the sample"]
  created_at timestamp
}


Table persons {
  name text [primary key]
  first_name text
  team_id text
  email email
  phone text
}


Table teams {
  name text [primary key]
  adress text [note: 'Adress of the labs']
}

Table synthesis_methods {
  name text [primary key]
}

Table characterization_methods {
  name text [primary key]
}




// un même échantillon peut être manufacturé par plusieurs méthodes différentes (exp, in silico)
//    a réfléchir. Une autre façon de faire serait de définir un champ "in silico version"
Ref :  samples.synthesis_id > synthesis.id

// many-to-one. Un même sample peut être analysé par plusieurs méthodes
Ref: characterizations.sample > samples.name

// un seul manufacturer par sample
Ref: synthesis.manufacturer > persons.name 
Ref: characterizations.operator > persons.name
Ref: synthesis.method > synthesis_methods.name
Ref: characterizations.method > characterization_methods.name

Ref: teams.name - persons.team_id



Records teams (name,adress) {
	'IPCMS','Strasbourg'
	'SOLEIL','Saclay'
	'LCMCP','Paris Sorbonne Université'
	'ITODYS','Université Paris Cité'
	'IFP Energies Nouvelles','Solaize'
}

Records persons(name,first_name,team_id) {
	'Bulou','Hervé','IPCMS'
	'Goyhenex','Christine','IPCMS'
	'Ersen','Ovidiu','IPCMS'
	'De Marco','Maria','IPCMS'
	'Takoutsin','Mikael','IPCMS'
	'Mirgot','Nathan','IPCMS'
	'Briois','Valérie','SOLEIL'
	'Fonda','Emiliano','SOLEIL'
	'Campolucci','Marta','LCMCP'
	'Faustini','Marco','LCMCP'
	'Ishiki','Nicolas','ITODYS'
	'Kanoufi','Frédéric','ITODYS'
	'Peron','Jennifer','ITODYS'
	'Charlety','Jean','IFP Energies Nouvelles'
}

Records synthesis_methods(name){
  "spray drying"
  "in silico"
}

Records characterization_methods(name){
  'TEM'
  'XAS'
  'Cyclic Voltammetry'  
}

Records synthesis(id,method,manufacturer,created_at){
	0,'in silico','Bulou','2025-09-09'
	1,'in silico','Bulou','2025-09-10'
	2,'spray drying','Campolucci','2025-06-10'	
	3,'spray drying','Campolucci','2025-07-15'	
}

Records samples(name,synthesis_id){
  'CuRuPdIrPt_0',2
  'CuRuPdIrPt_1',3
  'RhIr_0',1
  'NiRuIr_0',2
}


Records characterizations(id,method,operator,created_at,sample){
	0,'TEM','Mirgot','2026-02-19','CuRuPdIrPt_0'
	1,'XAS','Briois','2025-11-12','CuRuPdIrPt_0'
	2,'Cyclic Voltammetry','Ishiki','2025-10-04','CuRuPdIrPt_0'
	3,'TEM','Mirgot','2026-01-29','CuRuPdIrPt_1'
	4,'XAS','Briois','2025-11-09','CuRuPdIrPt_1'
}
    """
    
    # Sauvegarder l'exemple
    with open('example.dbml', 'w', encoding='utf-8') as f:
        f.write(example_dbml)
    
    # 2. Convertir DBML -> MongoDB JSON
    converter = DBMLToMongoConverter('example.dbml')
    converter.parse_dbml()
    
    # Optionnel: Ajouter des documents embarqués
    # converter.add_embedded_documents()
    
    # 3. Export en un seul fichier
    converter.export_to_json('mongodb_schema.json')
    
    # 4. Export en fichiers séparés (une collection = un fichier)
    converter.export_to_separate_files('mongodb_collections')
    
    print("\n✨ Conversion terminée!")
    print("📁 Fichiers générés:")
    print("   - mongodb_schema.json (schéma complet)")
    print("   - mongodb_collections/ (collections séparées)")


if __name__ == "__main__":
    main()
