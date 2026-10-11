import pytest
import slabsite

@pytest.mark.parametrize("show,env,expected", [(True,None,0),(False,"1",0),(True,"0",0),(True,"1",1)])
def test_view_is_explicit(monkeypatch, show, env, expected):
    calls = []
    monkeypatch.setattr(slabsite, "view", lambda atoms: calls.append(atoms))
    if env is None:
        monkeypatch.delenv("SLABSITE_SHOW_VORONOI", raising=False)
    else:
        monkeypatch.setenv("SLABSITE_SHOW_VORONOI", env)
    atoms = object()
    slabsite._maybe_view(atoms, show)
    assert calls == [atoms] * expected
