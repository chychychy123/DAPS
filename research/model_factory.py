"""Select the DAPS or CLIP-only ablation implementation."""
def build_model(config):
    from research.protocol import validate_attribute_resource
    validate_attribute_resource(config)
    if config.get("dino_used", True):
        from research.model import ResearchModel
    else:
        from supplementary.no_dino_model import ResearchModel
    return ResearchModel(config)
