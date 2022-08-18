=================
sox使用
=================

wav转pcm
===============

.. code-block:: shell

    sox aa.wav --bits 16 --encoding signed-integer --endian little bb.raw


修改采样率
=================
.. code-block:: shell

    sox a.wav -r 16000 b.wav
