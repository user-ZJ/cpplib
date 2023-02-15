#include <iostream>

// 定义颜色枚举类型，表示红黑树节点的颜色
enum Color { RED, BLACK };

// 红黑树节点模板类
template <typename T>
struct TreeNode {
  T key;             // 关键字值
  Color color;       // 颜色（红或黑）
  TreeNode *left;    // 左孩子节点
  TreeNode *right;   // 右孩子节点
  TreeNode *parent;  // 父节点

  // 构造函数
  TreeNode(T value, Color c, TreeNode *l, TreeNode *r, TreeNode *p) :
    key(value), color(c), left(l), right(r), parent(p) {}
};

// 红黑树模板类
template <typename T>
class RedBlackTree {
 private:
  TreeNode<T> *root;

  // 左旋
  void LeftRotate(TreeNode<T> *&root, TreeNode<T> *&pt) {
    TreeNode<T> *pt_right = pt->right;

    // 将pt的右孩子的左孩子作为pt的右孩子
    pt->right = pt_right->left;

    if (pt->right != NULL) pt->right->parent = pt;

    // 将pt_right的父节点设为pt的父节点
    pt_right->parent = pt->parent;

    // 如果pt是根节点，则将pt_right设为根节点
    if (pt->parent == NULL) root = pt_right;

    // 如果pt是其父节点的左孩子，则将pt_right设为其父节点的左孩子
    else if (pt == pt->parent->left)
      pt->parent->left = pt_right;

    // 如果pt是其父节点的右孩子，则将pt_right设为其父节点的右孩子
    else
      pt->parent->right = pt_right;
    // 将pt作为pt_right的左孩子
    pt_right->left = pt;
    pt->parent = pt_right;
  }

  // 右旋
  void RightRotate(TreeNode<T> *&root, TreeNode<T> *&pt) {
    TreeNode<T> *pt_left = pt->left;

    // 将pt的左孩子的右孩子作为pt的左孩子
    pt->left = pt_left->right;

    if (pt->left != NULL) pt->left->parent = pt;

    // 将pt_left的父节点设为pt的父节点
    pt_left->parent = pt->parent;

    // 如果pt是根节点，则将pt_left设为根节点
    if (pt->parent == NULL) root = pt_left;

    // 如果pt是其父节点的左孩子，则将pt_left设为其父节点的左孩子
    else if (pt == pt->parent->left)
      pt->parent->left = pt_left;

    // 如果pt是其父节点的右孩子，则将pt_left设为其父节点的右孩子
    else
      pt->parent->right = pt_left;

    // 将pt作为pt_left的右孩子
    pt_left->right = pt;
    pt->parent = pt_left;
  }

  // 插入修正函数，使得插入后仍然保持红黑树的性质
  void InsertFixUp(TreeNode<T> *&root, TreeNode<T> *&pt) {
    TreeNode<T> *parent_pt = NULL;
    TreeNode<T> *grand_parent_pt = NULL;

    // 循环执行直到没有父节点或者父节点是黑色
    while ((pt != root) && (pt->color != BLACK) && (pt->parent->color == RED)) {
      parent_pt = pt->parent;
      grand_parent_pt = pt->parent->parent;

      // 如果父节点是祖父节点的左孩子
      if (parent_pt == grand_parent_pt->left) {
        TreeNode<T> *uncle_pt = grand_parent_pt->right;

        // case 1: 叔叔节点是红色
        if ((uncle_pt != NULL) && (uncle_pt->color == RED)) {
          grand_parent_pt->color = RED;
          parent_pt->color = BLACK;
          uncle_pt->color = BLACK;
          pt = grand_parent_pt;
        } else {
          // case 2: 叔叔节点是黑色，且当前节点是右孩子
          if (pt == parent_pt->right) {
            LeftRotate(root, parent_pt);
            pt = parent_pt;
            parent_pt = pt->parent;
          }

          // case 3: 叔叔节点是黑色，且当前节点是左孩子
          parent_pt->color = BLACK;
          grand_parent_pt->color = RED;
          RightRotate(root, grand_parent_pt);
        }
      }

      // 如果父节点是祖父节点的右孩子
      else {
        TreeNode<T> *uncle_pt = grand_parent_pt->left;

        // case 1: 叔叔节点是红色
        if ((uncle_pt != NULL) && (uncle_pt->color == RED)) {
          grand_parent_pt->color = RED;
          parent_pt->color = BLACK;
          uncle_pt->color = BLACK;
          pt = grand_parent_pt;
        } else {
          // case 2: 叔叔节点是黑色，且当前节点是左孩子
          if (pt == parent_pt->left) {
            RightRotate(root, parent_pt);
            pt = parent_pt;
            parent_pt = pt->parent;
          }

          // case 3: 叔叔节点是黑色，且当前节点是右孩子
          parent_pt->color = BLACK;
          grand_parent_pt->color = RED;
          LeftRotate(root, grand_parent_pt);
        }
      }
    }

    // 将根节点设为黑色
    root->color = BLACK;
  }

  // 插入节点，并进行修正
  void InsertNode(TreeNode<T> *&root, TreeNode<T> *&pt) {
    TreeNode<T> *y = NULL;
    TreeNode<T> *x = root;

    // 找到合适的插入位置
    while (x != NULL) {
      y = x;
      if (pt->key < x->key)
        x = x->left;
      else
        x = x->right;
    }
    pt->parent = y;

    // 将新节点插入到合适的位置
    if (y == NULL)
      root = pt;
    else if (pt->key < y->key)
      y->left = pt;
    else
      y->right = pt;

    // 将新节点的颜色设为红色，并进行修正
    pt->color = RED;
    InsertFixUp(root, pt);
  }

  // 中序遍历
  void InOrderTraversal(TreeNode<T> *root) {
    if (root == NULL) return;
    InOrderTraversal(root->left);
    cout << root->key << " ";
    InOrderTraversal(root->right);
  }
};

int main() {
  RBTree<int> rbt;

  TreeNode<int> *root = NULL;

  // 插入若干个节点
  rbt.InsertNode(root, new TreeNode<int>(7));
  rbt.InsertNode(root, new TreeNode<int>(6));
  rbt.InsertNode(root, new TreeNode<int>(5));
  rbt.InsertNode(root, new TreeNode<int>(4));
  rbt.InsertNode(root, new TreeNode<int>(3));
  rbt.InsertNode(root, new TreeNode<int>(2));
  rbt.InsertNode(root, new TreeNode<int>(1));

  // 中序遍历
  cout << "InOrder Traversal: ";
  rbt.InOrderTraversal(root);
  cout << endl;

  return 0;
}
