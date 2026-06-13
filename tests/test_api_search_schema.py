from __future__ import annotations

from apps.api.schemas import SearchRequest


def test_image_search_accepts_image_path_alias() -> None:
    request = SearchRequest(
        search_type="image",
        image_path="data/external/hm/raw/images/052/0528568002.jpg",
        top_k=3,
    )

    assert request.image_url == request.image_path
