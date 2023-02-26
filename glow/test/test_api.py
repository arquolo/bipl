from glow.api import env


def test_context():
    assert env == {}

    with env(a=1):
        assert env == {'a': 1}
        with env(a=2, b=3):
            assert env == {'a': 1, 'b': 3}
        assert env == {'a': 1}

    assert env == {}


def test_deco():
    @env(a=2, b=3)
    def fn_1():
        return {**env}

    assert env == {}
    assert fn_1() == {'a': 2, 'b': 3}
    assert env == {}

    @env(a=1)
    def fn_2():
        return fn_1()

    assert fn_2() == {'a': 1, 'b': 3}
    assert env == {}


def test_mix():
    @env(a=2, b=3)
    def fn_1():
        return {**env}

    @env(a=1)
    def fn_2():
        return fn_1()

    with env(a=1):
        assert env == {'a': 1}
        assert fn_1() == {'a': 1, 'b': 3}

    with env(b=4):
        assert fn_2() == {'a': 1, 'b': 4}

    assert env == {}
