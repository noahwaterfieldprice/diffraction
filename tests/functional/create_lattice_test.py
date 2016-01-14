from diffraction import DirectLattice


class TestCreatingDirectLatticeFromSequence:
    def test_can_create_from_sequence(self):
        lattice = DirectLattice([4.99, 4.99, 17.002, 90, 90, 120])

        assert lattice.a == 4.99
        assert lattice.b == 4.99
        assert lattice.c == 17.002
        assert lattice.alpha == 90
        assert lattice.beta == 90
        assert lattice.gamma == 120
