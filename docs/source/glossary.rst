Glossary
========

.. glossary::
   :sorted:

   CIF
    A CIF (Crystallographic Information File) is a file conforming to the CIF
    syntax which contains information on a crystallographic experiment. A
    detailed results (or similar scientific content); or descriptions of the
    data specification of the CIF syntax can be found here
    http://www.iucr.org/resources/cif/spec/version1.1 ).

   data item
   data items
       A data item is a specific piece of information defined by a
       :term:`data name` and an associated :term:`data value`.

   data name
   data names
      A data name is a case-insensitive identifier (a string of characters
      beginning with an underscore character) of the content of an associated
      :term:`data value`.

   data value
   data values
      A data value is a string of characters representing a particular item of
      information e.g. a single numerical value; a letter, word or phrase, or
      extended text.

   data block
   data blocks
      A data block is the highest-level component of a :term:`CIF`, containing
      :term:`data items`. A data block is identified by a
      :term:`data block header` at the beginning.

   data block header
      A data block header is an string that identifies a :term:`data block`,
      beginning with the case-insensitive reserved characters `data_`. Bounded
      by whitespace, it precedes the :term:`data block` on a line of it's own.

   lattice parameters
      The dimensions describing :term:`unit cell` of the crystal structure.

   loop
      sadsdoajidjasoidj a

   metric tensor
      In crystallography, a metric tensor is used to calculate distances in both
      direct and reciprocal space. Consider two vectors in a Euclidean space
      with a basis :math:`\mathbf{e}_i`, :math:`\mathbf{u} = u^i \mathbf{e}_i`
      and :math:`\mathbf{v} = v^j \mathbf{e}_j`. The scalar product is then
      :math:`\mathbf{u} \cdot \mathbf{v} = u^i v^j g_{ij}` where :math:`g_{ij}
      = \mathbf{e}_i \cdot \mathbf{e}_j` is the metric tensor.
      
      For a direct lattice with lattice vectors :math:`\mathbf{a}`,
      :math:`\mathbf{b}` and :math:`\mathbf{c}` the metric tensor in matrix form
      is given by:

         .. math::
            \mathbf{G} = \left(
                            \begin{matrix}
                                \mathbf{a} \cdot \mathbf{a} & \mathbf{a} \cdot \mathbf{b} & \mathbf{a} \cdot \mathbf{c}\\
                                \mathbf{b} \cdot \mathbf{a} & \mathbf{b} \cdot \mathbf{b} & \mathbf{b} \cdot \mathbf{c}\\
                                \mathbf{v} \cdot \mathbf{a} & \mathbf{c} \cdot \mathbf{b} & \mathbf{c} \cdot \mathbf{c}\\
                            \end{matrix}
                         \right)

   semicolon data item
      A semicolon data item is a :term:`data item` where the :term:`data value`
      is a :term:`semicolon text field`.

   semicolon text field
      A semicolon text field is a :term:`data value` delimited by two
      semicolons located at the start of a line. The field may contain
      semicolons so long as they do not immediately follow a newline character.

   space group
      The group of symmetry operators that describe the symmetry operations of
      the crystal lattice.

   Unit cell
      The repeat unit of the crystal structure