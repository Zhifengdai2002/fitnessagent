from __future__ import annotations

from agent.rag.documents import (
    LOCAL_FOOD_FALLBACK_LIMIT,
    build_exercise_documents,
    build_food_documents,
    load_local_food_fallback_source,
)
from agent.rag.retriever import rebuild_food_index, retrieve_exercises, retrieve_foods
from agent.tools.exercise_tool import get_exercise_by_name
from agent.tools import food_tool
from agent.tools.food_tool import calculate_food_macros, find_foods, get_food_by_name


def test_exercise_rag_seed_documents_are_retrievable_tool_sources() -> None:
    documents = build_exercise_documents()

    assert any(doc["metadata"]["source"] == "curated_rag" for doc in documents)
    assert get_exercise_by_name("Machine Chest Press")


def test_food_rag_documents_expand_food_knowledge() -> None:
    documents = build_food_documents()
    names = {doc["metadata"]["name"] for doc in documents}

    assert len(documents) > 16
    assert "Turkey Breast" in names
    assert "Green Beans" in names


def test_food_rag_retrieval_respects_category_and_allergens() -> None:
    rebuild_food_index()

    foods = retrieve_foods(
        query="lean high protein low calorie lunch",
        category="protein",
        excluded_allergens=["fish", "shellfish"],
        min_protein_g=10,
        limit=5,
    )

    assert foods
    assert all(food["category"] == "protein" for food in foods)
    assert all("fish" not in food.get("allergens", []) for food in foods)
    assert any(food["name"] in {"Turkey Breast", "Chicken Breast"} for food in foods)


def test_food_rag_uses_learned_avoidances() -> None:
    rebuild_food_index()

    foods = retrieve_foods(
        query="lean high protein low calorie lunch",
        category="protein",
        min_protein_g=10,
        learned_preferences={"avoided_foods": ["Turkey Breast", "Tuna", "Shrimp"]},
        limit=8,
    )
    names = {food["name"] for food in foods}

    assert "Turkey Breast" not in names
    assert "Tuna" not in names
    assert "Shrimp" not in names
    assert foods


def test_find_foods_prefers_rag_seed_candidates_and_macro_lookup_works() -> None:
    foods = find_foods(category="vegetable", excluded_allergens=[], limit=6)
    names = {food["name"] for food in foods}

    assert "Green Beans" in names or "Bell Pepper" in names or "Zucchini" in names
    assert get_food_by_name("Turkey Breast")
    macros = calculate_food_macros("rag_turkey_breast", 150)
    assert macros["protein_g"] > 40


def test_find_foods_passes_learned_preferences_to_rag() -> None:
    foods = find_foods(
        category="vegetable",
        learned_preferences={"avoided_foods": ["Broccoli", "Spinach"]},
        limit=8,
    )
    names = {food["name"] for food in foods}

    assert "Broccoli" not in names
    assert "Spinach" not in names
    assert {"Green Beans", "Bell Pepper", "Zucchini"}.intersection(names)


def test_food_rag_keeps_local_candidates_as_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.rag.retriever._search_food_documents",
        lambda query, limit: [
            {
                "score": 0.99,
                "document": {
                    "metadata": {
                        "name": "Local Protein",
                        "source": "local",
                        "category": "protein",
                        "protein_g": 30,
                        "allergens": [],
                    },
                    "raw": {
                        "name": "Local Protein",
                        "source": "local",
                        "category": "protein",
                        "protein_g": 30,
                        "allergens": [],
                    },
                },
            },
            {
                "score": 0.4,
                "document": {
                    "metadata": {
                        "name": "Curated Protein",
                        "source": "curated_rag",
                        "category": "protein",
                        "protein_g": 28,
                        "allergens": [],
                    },
                    "raw": {
                        "name": "Curated Protein",
                        "source": "curated_rag",
                        "category": "protein",
                        "protein_g": 28,
                        "allergens": [],
                    },
                },
            },
        ],
    )

    foods = retrieve_foods(
        query="lean high protein",
        category="protein",
        min_protein_g=10,
        limit=2,
    )

    assert [food["name"] for food in foods] == ["Curated Protein", "Local Protein"]
    assert foods[1]["source"] == "local_fallback"


def test_exercise_rag_dedupes_near_identical_names(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent.rag.retriever._search_exercise_documents",
        lambda query, limit: [
            {
                "score": 1.0,
                "document": {
                    "metadata": {
                        "name": "Machine Chest Press Exercise",
                        "source": "wger",
                        "focus_tags": ["upper_chest_arms"],
                        "difficulty": "beginner",
                    },
                    "raw": {
                        "name": "Machine Chest Press Exercise",
                        "source": "wger",
                        "focus_tags": ["upper_chest_arms"],
                        "difficulty": "beginner",
                    },
                },
            },
            {
                "score": 0.99,
                "document": {
                    "metadata": {
                        "name": "Machine Chest Press",
                        "source": "curated_rag",
                        "focus_tags": ["upper_chest_arms"],
                        "difficulty": "beginner",
                    },
                    "raw": {
                        "name": "Machine Chest Press",
                        "source": "curated_rag",
                        "focus_tags": ["upper_chest_arms"],
                        "difficulty": "beginner",
                    },
                },
            },
            {
                "score": 0.8,
                "document": {
                    "metadata": {
                        "name": "Incline Push-Up",
                        "source": "curated_rag",
                        "focus_tags": ["upper_chest_arms"],
                        "difficulty": "beginner",
                    },
                    "raw": {
                        "name": "Incline Push-Up",
                        "source": "curated_rag",
                        "focus_tags": ["upper_chest_arms"],
                        "difficulty": "beginner",
                    },
                },
            },
        ],
    )

    exercises = retrieve_exercises(query="upper chest", focus="upper_chest_arms", level="beginner", limit=2)
    names = [exercise["name"] for exercise in exercises]

    assert len(names) == 2
    assert "Machine Chest Press" in names
    assert "Machine Chest Press Exercise" not in names
    assert "Incline Push-Up" in names


def test_find_foods_uses_local_only_after_primary_sources(monkeypatch) -> None:
    monkeypatch.setattr(
        food_tool,
        "_retrieve_rag_foods",
        lambda **_: [],
    )
    monkeypatch.setattr(
        food_tool,
        "load_food_db",
        lambda: [
            {
                "id": "local_chicken",
                "name": "Local Chicken",
                "category": "protein",
                "protein_g": 31,
                "calories_per_100g": 165,
                "carbs_g": 0,
                "fat_g": 4,
                "fiber_g": 0,
                "diet_tags": ["high_protein"],
                "allergens": [],
                "source": "local",
            },
            {
                "id": "rag_turkey",
                "name": "RAG Turkey",
                "category": "protein",
                "protein_g": 29,
                "calories_per_100g": 135,
                "carbs_g": 0,
                "fat_g": 2,
                "fiber_g": 0,
                "diet_tags": ["high_protein"],
                "allergens": [],
                "source": "curated_rag",
            },
        ],
    )

    foods = food_tool.find_foods(category="protein", min_protein_g=10, limit=1)

    assert [food["name"] for food in foods] == ["RAG Turkey"]


def test_food_documents_keep_local_json_as_small_fallback_only() -> None:
    fallback = load_local_food_fallback_source()
    documents = build_food_documents()
    local_documents = [doc for doc in documents if doc["metadata"]["source"] == "local_fallback"]
    primary_sources = {doc["metadata"]["source"] for doc in documents if doc["metadata"]["source"] != "local_fallback"}
    fallback_names = [item["name"] for item in fallback]

    assert len(fallback) == LOCAL_FOOD_FALLBACK_LIMIT
    assert len(local_documents) <= LOCAL_FOOD_FALLBACK_LIMIT
    assert primary_sources == {"curated_rag"}
    assert fallback_names[:7] == [
        "Chicken Breast",
        "Egg",
        "Firm Tofu",
        "Brown Rice, Cooked",
        "Sweet Potato",
        "Broccoli",
        "Apple",
    ]


def test_legacy_local_foods_are_exact_lookup_only() -> None:
    foods = food_tool.load_food_db()
    food_names = {food["name"] for food in foods}

    assert "Greek Yogurt, Nonfat" not in food_names
    legacy = food_tool.get_food_by_name("Greek Yogurt, Nonfat")
    assert legacy
    assert legacy["source"] == "local_fallback"
