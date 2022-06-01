import java.util.concurrent.Semaphore;

public class Main {

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    int a = 0;
    String str = "123";

    synchronized void add() {
        a++;
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null){
            return null;
        }
        if (root.val == p.val || root.val == q.val){
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null){
            return root;
        } else if(left != null && right == null){
            return left;
        }
        return right;
    }

    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        TreeNode smaller;
        TreeNode bigger;
        if (p.val> q.val){
            smaller = q;
            bigger = p;
        } else {
            smaller = p;
            bigger = q;
        }
        if (root.val > smaller.val && root.val < bigger.val || (root.val == p.val) || (root.val == q.val)){
            return root;
        } else if(root.val > smaller.val && root.left != null){
            return lowestCommonAncestor(root.left, p, q);
        } else if(root.val < bigger.val && root.right != null){
            return lowestCommonAncestor(root.right, p, q);
        }
        return null;
    }

    void testString() {
        String str2 = new String("str01");
        String str1 = "str01";
        str2.intern();
        System.out.println(str2 == str1);

        ITest test = new ITest() {
            @Override
            public void test() {
                String b = str;
                str = "1234";
            }
        };
    }

    interface ITest {
        void test();
    }


    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    /**
     * 两个链表的公共节点
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        int aLength = 0;
        int bLength = 0;
        ListNode a = headA;
        ListNode b = headB;
        while (a != null || b != null) {
            if (a != null) {
                aLength++;
                a = a.next;
            }
            if (b != null) {
                bLength++;
                b = b.next;
            }
        }

        ListNode a2 = headA;
        ListNode b2 = headB;
        if (aLength > bLength) {
            for (int i = 0; i < aLength - bLength; ++i) {
                a2 = a2.next;
            }
        } else {
            for (int i = 0; i < bLength - aLength; ++i) {
                b2 = b2.next;
            }
        }

        while (a2 != null && b2 != null) {
            if (a2.val == b2.val) {
                return a2;
            }
            a2 = a2.next;
            b2 = b2.next;
        }
        return null;
    }
}
