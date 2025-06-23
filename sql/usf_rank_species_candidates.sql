-- ==========================================================
-- Function: usf_rank_species_candidates
-- Purpose:  Rank species based on image vector similarity and
--           ecological location relevance.
--
-- Inputs:
--   lat       - Latitude of the observation
--   lon       - Longitude of the observation
--   embedding - 512-dim vector representing the image (pgvector)
--   category  - Optional species category filter (default: 'unknown')
--   top_n     - Number of top-ranked species to return (default: 5)
--
-- Outputs:
--   species           - Species scientific name
--   common_name       - Common name
--   image_path        - Path to reference image
--   distance          - Raw cosine distance between embeddings
--   location_boosted  - TRUE if species is known in the matching ecoregion
--   final_score       - Combined score: 60% similarity, 40% location relevance
--
-- Dependencies:
--   - Function: public.get_ecoregion_by_coords(lat, lon)
--   - Table: wildlife.species_embedding
--   - Table: species_ecoregion (mapping species to ecoregions)
--
-- Example:
--   SELECT * FROM wildlife.usf_rank_species_candidates(44.6, -110.5, '[...]'::vector);
--
-- Notes:
--   - Uses pgvector `<->` for cosine distance
--   - Boosts results for species present in the local ecoregion
-- ==========================================================

CREATE OR REPLACE FUNCTION wildlife.usf_rank_species_candidates(
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    embedding VECTOR,
    category TEXT DEFAULT 'unknown',
    top_n INTEGER DEFAULT 5
)
RETURNS TABLE (
    species TEXT,
    common_name TEXT,
    image_path TEXT,
    distance DOUBLE PRECISION,
    location_boosted BOOLEAN,
    final_score DOUBLE PRECISION
)
LANGUAGE plpgsql
COST 100
VOLATILE
PARALLEL UNSAFE
AS $$
BEGIN
    RETURN QUERY

    WITH matched_ecoregion AS (
        SELECT eco_code
        FROM public.get_ecoregion_by_coords(lat, lon)
    ),

    loc_species AS (
        SELECT DISTINCT se.common_name
        FROM species_ecoregion se
        JOIN matched_ecoregion me ON se.eco_code = me.eco_code
    ),

    ranked AS (
        SELECT
            se.species,
            se.common_name,
            se.image_path,
            se.image_embedding <-> embedding AS distance,
            (se.common_name IN (SELECT ls.common_name FROM loc_species ls)) AS location_boosted,
            -- Final Score: Weighted combination of similarity and location relevance
            (0.6 * (1 - (se.image_embedding <-> embedding))) +
            (0.4 * CASE WHEN se.common_name IN (SELECT ls.common_name FROM loc_species ls) THEN 1 ELSE 0 END) AS final_score
        FROM wildlife.species_embedding se
        WHERE se.image_embedding IS NOT NULL
    )

    SELECT *
    FROM ranked
    ORDER BY final_score DESC
    LIMIT top_n;

END;
$$;

ALTER FUNCTION wildlife.usf_rank_species_candidates(
    DOUBLE PRECISION, DOUBLE PRECISION, VECTOR, TEXT, INTEGER
) OWNER TO wildlife_user;
