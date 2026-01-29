#!/usr/bin/env python3
"""
Test script for multi-model semantic analysis
"""

import sys
import os
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from multi_model_semantic import analyze_query_multi_model, get_analyzer
    print("✅ Multi-model semantic analyzer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import multi-model semantic analyzer: {e}")
    print("   Make sure sentence-transformers is installed: pip install sentence-transformers")
    sys.exit(1)

# Test queries
test_queries = [
    "Find customers in California with transactions over $1000",
    "Show me top 10 customers by transaction volume",
    "List ACH transactions in Texas",
    "Find disputes for customer CUST12345",
    "Show customers in ZIP code 75001 with credit card transactions",
]

def test_analysis(query: str):
    """Test semantic analysis for a query."""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    try:
        result = analyze_query_multi_model(query, parallel=True)
        
        print(f"\n📊 Models Analyzed: {result['models_analyzed']}")
        
        # Show consensus tags
        consensus_tags = result['aggregated']['consensus_tags']
        if consensus_tags:
            print(f"\n🎯 Consensus Tags (detected by 2+ models):")
            for tag in consensus_tags:
                print(f"   • {tag['label']}: {tag['value']} (confidence: {tag['confidence']:.2f}, models: {tag['detected_by_models']})")
        
        # Show summary
        summary = result['summary']
        if summary.get('most_common_entity'):
            print(f"\n📈 Summary:")
            print(f"   • Most Common Entity: {summary['most_common_entity']}")
        if summary.get('most_common_action'):
            print(f"   • Most Common Action: {summary['most_common_action']}")
        if summary.get('most_common_tag_type'):
            print(f"   • Most Common Tag Type: {summary['most_common_tag_type']}")
        
        # Show tag counts
        tag_counts = result['aggregated']['tag_counts']
        if tag_counts:
            print(f"\n🏷️  Tag Counts:")
            for tag_type, count in tag_counts.items():
                print(f"   • {tag_type}: {count}")
        
        # Show entity counts
        entity_counts = result['aggregated']['entity_counts']
        if entity_counts:
            print(f"\n👥 Entity Counts:")
            for entity, count in entity_counts.items():
                print(f"   • {entity}: {count}")
        
        # Show a sample model result
        if result['models']:
            first_model = list(result['models'].keys())[0]
            model_result = result['models'][first_model]
            if 'error' not in model_result:
                print(f"\n🔍 Sample Model Analysis ({first_model}):")
                print(f"   • Tags: {len(model_result.get('tags', []))}")
                print(f"   • Entities: {', '.join(model_result.get('detected_entities', []))}")
                print(f"   • Actions: {', '.join(model_result.get('detected_actions', []))}")
                if model_result.get('embedding_stats'):
                    stats = model_result['embedding_stats']
                    print(f"   • Embedding Dimension: {stats.get('dimension')}")
                    print(f"   • Embedding Norm: {stats.get('norm', 0):.2f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error analyzing query: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run tests."""
    print("🚀 Multi-Model Semantic Analysis Test")
    print("="*80)
    
    # Check if analyzer can be initialized
    try:
        analyzer = get_analyzer()
        model_info = analyzer.get_model_info()
        print(f"\n✅ Analyzer initialized with {len(model_info)} models:")
        for model_name, info in model_info.items():
            print(f"   • {model_name} (dimension: {info['dimension']})")
    except Exception as e:
        print(f"❌ Failed to initialize analyzer: {e}")
        sys.exit(1)
    
    # Test each query
    results = []
    for query in test_queries:
        result = test_analysis(query)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Test Complete: {len(results)}/{len(test_queries)} queries analyzed successfully")
    print(f"{'='*80}")
    
    # Optionally save results to JSON
    if results:
        output_file = "multi_model_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
