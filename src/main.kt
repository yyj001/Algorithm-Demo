import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap

@ExperimentalCoroutinesApi
@ExperimentalStdlibApi
suspend fun main() {
  print(Solution().print().toString())
}


class Solution{
  fun print():Any{
    return reconstructQueue(arrayOf(intArrayOf(7,0), intArrayOf(4,4), intArrayOf(7,1),intArrayOf(5,0),intArrayOf(6,1),intArrayOf(5,2)))
  }

  fun reconstructQueue(people: Array<IntArray>): Array<IntArray> {
    people.sortWith(compareBy({it[0]}, {-it[1]}))
    val result = Array(people.size){ intArrayOf(-1,-1)}
    for (p in people){
      var count = -1
      var index = 0
      while (index < people.size){
        if (result[index][0] < 0){
          count++
        }
        if (count >= p[1]){
          break
        }
        index++
      }
      result[index--] = p
    }
    return result
  }

  fun combinationSum(candidates: IntArray, target: Int): List<List<Int>> {
    if (candidates.isEmpty()) return emptyList()
    val result = ArrayList<List<Int>>()
    getSample(candidates, target, 0, 0, LinkedList<Int>(), result)
    return result
  }

  fun getSample(candidates: IntArray, target: Int, index: Int, sum: Int, list: LinkedList<Int>, result: ArrayList<List<Int>>){
    if (sum > target){
      return
    }
    if (sum == target){
      result.add(list.toList())
      return
    }
    for (i in index until candidates.size){
      list.addLast(candidates[i])
      getSample(candidates, target, i, sum + candidates[i], list, result)
      list.removeLast()
    }
  }

  fun rotate(matrix: Array<IntArray>): Unit {
    for (layerNum in 0 until matrix.size / 2){
      var row = layerNum
      var column = layerNum
      val rotateCount = matrix.size - layerNum * 2 - 1 // 3个数
      for (i in 0 until rotateCount){
        var tempRow = row
        var tempColumn = column + i
        var lastValue = matrix[tempRow][tempColumn]
        repeat(4){
          val newRow = tempColumn
          val newColumn =  matrix.size - 1 - tempRow
          val tempValue = matrix[newRow][newColumn]
          matrix[newRow][newColumn] = lastValue
          lastValue = tempValue
          tempRow = newRow
          tempColumn = newColumn
        }
      }
    }
  }

  fun permute(nums: IntArray): List<List<Int>> {
    var result = ArrayList<List<Int>>()
    result.add(ArrayList())
    for (num in nums){
      val tempResult = ArrayList<List<Int>>()
      for (list in result){
        for (i in 0..list.size){
          val tempList = ArrayList<Int>()
          tempList.addAll(list)
          tempList.add(i, num)
          tempResult.add(tempList)
        }
      }
      result = tempResult
    }
    return result
  }

  fun subsets(nums: IntArray): List<List<Int>> {
    val result = arrayListOf<List<Int>>()
    result.add(arrayListOf())
    genSubList(nums, 0, result)
    return result
  }

  fun genSubList(nums: IntArray, index: Int, result: ArrayList<List<Int>>){
    if (index >= nums.size) return
    val current = nums[index]
    genSubList(nums, index + 1, result)
    val tempRes = arrayListOf<List<Int>>()
    result.forEach {
      val tempArray = arrayListOf(current)
      tempArray.addAll(it)
      tempRes.add(tempArray)
    }
    result.addAll(tempRes)
  }

  fun trap(height: IntArray): Int {
    val maxRights = IntArray(height.size)
    var maxRight = 0
    for (i in height.size - 1 downTo 0){
      maxRight = Math.max(maxRight, height[i])
      maxRights[i] = maxRight
    }

    var maxLeft = 0
    var sum = 0
    for (i in height.indices){
      val max = Math.min(maxLeft, maxRights[i])
      if (max > height[i]){
        sum += max - height[i]
      }
      maxLeft = Math.max(maxLeft, height[i])
    }
    return sum
  }

  fun subarraySum(nums: IntArray, k: Int): Int {
    val map = hashMapOf<Int, Int>()
    var count = 0
    var sum = 0
    map[0] = 1
    for (num in nums){
      sum += num
      if (map.containsKey(sum - k)){
        count += map.get(sum - k)!!
      }
      map.put(sum, map.getOrDefault(sum, 0) + 1)
    }
    return count
  }


  fun dailyTemperatures(temperatures: IntArray): IntArray {
    val result = IntArray(temperatures.size){0}
    val stack = Stack<Int>()
    for (temp in temperatures.withIndex()){
      if (stack.isEmpty()){
        stack.push(temp.index)
      } else{
        while (stack.isNotEmpty()){
          val topIndex = stack.peek()
          val top = temperatures[topIndex]
          if (top < temp.value){
            result[topIndex] = temp.index - topIndex
            stack.pop()
          } else {
            break
          }
        }
        stack.push(temp.index)
      }
    }
    return result
  }

  fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
    var end: ListNode? = head
    for (i in 0 until n){
      end = end?.next
    }
    var start: ListNode? = head
    var pStart: ListNode? = null
    while (end != null){
      end = end.next
      pStart = start
      start = start?.next
    }
    if (pStart == null){
      return head?.next
    }
    pStart.next = start?.next
    return head
  }

  fun diameterOfBinaryTree(root: TreeNode?): Int {
    return getTreeDeep(root).second
  }

  fun getTreeDeep(root: TreeNode?): Pair<Int, Int>{
    if (root == null){
      return 0 to 0
    }
    val left = getTreeDeep(root.left)
    val right = getTreeDeep(root.right)
    val maxLength = Math.max(Math.max(left.second, right.second),left.first + right.first)
    return Math.max(left.first, right.first) + 1 to maxLength
  }

  fun findDisappearedNumbers(nums: IntArray): List<Int> {
    val result = arrayListOf<Int>()
    var targetPosition = 0
    for (i in nums.indices){
      var current = nums[i]
      if (current != i + 1){
        targetPosition = current - 1
        while (current != nums[targetPosition]){
          val temp = nums[current - 1]
          nums[current - 1] = current
          current = temp
          targetPosition = current - 1
        }
      }
    }
    for (i in nums.indices){
      if (i + 1 != nums[i]){
        result.add(i + 1)
      }
    }
    return result
  }

  fun moveZeroes(nums: IntArray): Unit {
    var i = 0
    var j = 0
    while(i < nums.size && j < nums.size){
      val ti = i
      for (index in ti until nums.size){
        if (nums[index] == 0) break
        i++
      }
      if (j < i) j = i
      val tj = j
      for (index in tj until nums.size){
        if (nums[index] != 0) break
        j++
      }
      if (i >= nums.size || j >= nums.size) break
      nums[i] = nums[j]
      nums[j] = 0
      i++
      j++
    }
  }

  fun isPalindrome(head: ListNode?): Boolean {
    if (head == null) return false
    var count = head.getSize()
    if (count == 1) return true
    val middle = (count + 1) / 2
    var middleP = head
    for (i in 0 until middle){
      middleP = middleP?.next
    }
    val reverseList = reverseList(middleP)
    var a = head
    var b = reverseList
    while(a != null && b != null){
      if (a.`val` != b.`val`) return false
      a = a.next
      b = b.next
    }
    return true
  }

  fun reverseList(head: ListNode?): ListNode? {
    if (head == null) return null
    if (head.next == null) return head
    val next = head.next
    val h = reverseList(head.next)
    head.next = null
    next?.next = head
    return h
  }

  fun majorityElement(nums: IntArray): Int {
    var num = 0
    var count = 0
    for (n in nums){
      if (n == num){
        count++
      } else {
        if (count==0){
          num = n
          count = 1
        } else{
          count--
        }
      }
    }
    return num
  }

  fun getIntersectionNode(headA:ListNode?, headB:ListNode?):ListNode? {
    if (headA == null || headB == null) return null
    val aSize = headA.getSize()
    val bSize = headB.getSize()
    var tempA = headA
    var tempB = headB
    if (aSize > bSize){
      for (i in 0 until aSize - bSize){
        tempA = tempA?.next
      }
    } else {
      for (i in 0 until bSize - aSize){
        tempB = tempB?.next
      }
    }
    while (tempA != null && tempB != null){
      if (tempA == tempB) return tempA
      tempA = tempA.next
      tempB = tempB.next

    }
    return null
  }

  fun ListNode?.getSize():Int{
    var temp:ListNode? = this
    var count = 0
    while (temp!=null){
      temp = temp.next
      count++
    }
    return count
  }

  fun isSymmetric(root: TreeNode?): Boolean {
    return checkTree(root?.left, root?.right)
  }

  fun checkTree(leftT: TreeNode?, rightT: TreeNode?): Boolean{
    if (leftT == null && rightT == null) return true
    if (leftT?.`val` != rightT?.`val`) return false
    return checkTree(leftT?.left, rightT?.right) && checkTree(leftT?.right, rightT?.left)
  }

  fun buildLink(intArray: IntArray): ListNode{
    return ListNode(intArray.first()).apply {
      var head:ListNode? = this
      for (i in 1 until intArray.size){
        head?.next = ListNode(intArray[i])
        head = head?.next
      }
    }
  }

  fun hasCycle(head: ListNode?): Boolean {
    if (head == null) return false
    var fast: ListNode? = head.next
    var slow: ListNode? = head
    while (fast != null && slow != null){
      if (fast == slow) return true
      fast = fast.next?.next
      slow = slow.next
    }
    return false
  }

  class MinStack() {

    val stack1 = Stack<Int>()
    val stack2 = Stack<Int>()

    fun push(`val`: Int) {
      stack1.push(`val`)
      if (stack2.isEmpty()){
        stack2.push(`val`)
      } else {
        val value = Math.min(`val`, stack2.peek())
        stack2.push(value)
      }
    }

    fun pop() {
      stack1.pop()
      stack2.pop()
    }

    fun top(): Int {
      return stack1.peek()
    }

    fun getMin(): Int {
      return stack2.peek()
    }
  }


  fun mergeTrees(root1: TreeNode?, root2: TreeNode?): TreeNode? {
    if (root1 == null && root2 == null ) return null
    var root = TreeNode((root1?.`val`?: 0) + (root2?.`val`?: 0) )
    root.left = mergeTrees(root1?.left, root2?.left)
    root.right = mergeTrees(root1?.right, root2?.right)
    return root
  }

  fun invertTree(root: TreeNode?): TreeNode? {
    root?.let{
      invertTree(it.left)
      invertTree(it.right)
      val temp = it.right
      it.right = it.left
      it.left = temp
    }
    return root
  }

  fun hammingDistance(x: Int, y: Int): Int {
    var z = x xor y
    var count = 0
    while (z > 0){
      if (z % 2 == 1){
        count++
      }
      z = z shr 1
    }
    return count
  }

  fun maxProfit(prices: IntArray): Int {
    var minValue = Int.MAX_VALUE
    var max = 0
    for (p in prices){
      max = Math.max(max, p - minValue)
      minValue = Math.min(p, minValue)
    }
    return max
  }

  fun inorderTraversal(root: TreeNode?): List<Int> {
    if (root == null) return emptyList()
    val result = arrayListOf<Int>()
    middleSearch(root, result)
    return result
  }

  private fun middleSearch(root: TreeNode, container: ArrayList<Int>){
    root.left?.let { middleSearch(it, container) }
    container.add(root.`val`)
    root.right?.let { middleSearch(it, container) }
  }



  /**
   * 三数之和
   */
  fun threeSum(nums: IntArray): List<List<Int>> {
    val result = HashSet<List<Int>>()
    val sorted = nums.sorted()
    val target = 0
    for (i in sorted.indices){
      findTarget(sorted, target - sorted[i], i + 1, sorted.size - 1, sorted[i], result)
    }
    return result.toList()
  }

  fun findTarget(array: List<Int>, sum: Int, s: Int, e: Int, first: Int, container: HashSet<List<Int>>){
    var start = s
    var end = e
    while (start < end){
      when{
        sum > array[start] + array[end] -> start++
        sum < array[start] + array[end] -> end--
        else -> {
          val r = arrayListOf(first, array[start], array[end])
          if (!container.contains(r)){
            container.add(r)
          }
          start++
        }
      }
    }
  }
}


/**
 * 水桶最大容量
 */
fun maxArea(height: IntArray): Int {
  if (height.size == 1 || height.isEmpty()) return 0
  var max = 0
  var sIndex = 0
  var eIndex = height.size - 1
  while (sIndex < eIndex){
    max = Math.max(Math.min(height[sIndex], height[eIndex]) * (eIndex - sIndex), max)
    if (height[sIndex] > height[eIndex]){
      eIndex--
    }else{
      sIndex++
    }
  }
  return max
}

/**
 * 最长回文子串
 */
fun longestPalindrome(s: String): String {
  var maxStr = s.firstOrNull()?.toString()?: ""
  for (index in s.indices){
    if (index == s.length - 1) continue
    if (s[index + 1] == s[index]){
      val max = getMaxStr(s, index, index + 1)
      if (max.length > maxStr.length){
        maxStr = max
      }
    }
    val max = getMaxStr(s, index, index)
    if (max.length > maxStr.length){
      maxStr = max
    }
  }
  return maxStr
}

fun getMaxStr(s: String, left: Int, right: Int): String{
  var l = left
  var r = right
  while (l >=0 && r < s.length){
    if (s[l] != s[r]) break
    l--
    r++
  }
  l++
  r--
  return s.substring(l, r + 1)
}

/**
 * 剑指 Offer 48. 最长不含重复字符的子字符串
 */
fun lengthOfLongestSubstring(s: String): Int {
  val map = hashMapOf<Char, Int>()
  var maxLength = 0
  var start = 0
  for (index in s.indices){
    val c = s.get(index)
    if (map.containsKey(c) && map.getOrDefault(c, Int.MAX_VALUE) >= start){
      start = map.getOrDefault(c, start) + 1
    } else {
      maxLength = Math.max(index - start + 1, maxLength)
    }
    map[c] = index
  }
  return maxLength
}

/**
 * 正序数组的中位数
 */
fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
  val target1 = (nums1.size + nums2.size + 1) / 2
  val num1 = findMedianSortedArrays(nums1, nums2, 0, 0, target1)
  val target2 = (nums1.size + nums2.size + 2) / 2
  val num2 = findMedianSortedArrays(nums1, nums2, 0, 0, target2)
  return (num1 + num2) / 2.0
}

fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray, start1: Int, start2: Int, k: Int): Int{
  if (start1 >= nums1.size){
    return nums2[start2 + k - 1]
  }
  if (start2 >= nums2.size){
    return nums1[start1 + k - 1]
  }
  if (k == 1){
    return Math.min(nums1[start1], nums2[start2])
  }

  val middle1 = if (start1 + k / 2 - 1 < nums1.size) nums1[start1 + k / 2 - 1] else Int.MAX_VALUE
  val middle2 = if (start2 + k / 2 - 1 < nums2.size) nums2[start2 + k / 2 - 1] else Int.MAX_VALUE

  return if (middle1 < middle2){
    findMedianSortedArrays(nums1, nums2, start1 + k/2, start2, k - k / 2)
  } else {
    findMedianSortedArrays(nums1, nums2, start1, start2 + k / 2, k - k / 2)
  }
}


//  fun findMedianSortedArrays2(nums1: IntArray, nums2: IntArray): Double {
//    val m = nums1.size
//    val n = nums2.size
//    val left = (m + n + 1) / 2
//    val right = (m + n + 2) / 2
//    return (findKth(nums1, 0, nums2, 0, left) + findKth(nums1, 0, nums2, 0, right)) / 2.0
//  }
//
//  //i: nums1的起始位置 j: nums2的起始位置
//  fun findKth(nums1: IntArray, i: Int, nums2: IntArray, j: Int, k: Int): Int {
//    if (i >= nums1.size) return nums2[j + k - 1] //nums1为空数组
//    if (j >= nums2.size) return nums1[i + k - 1] //nums2为空数组
//    if (k == 1) {
//      return Math.min(nums1[i], nums2[j])
//    }
//    val midVal1 = if (i + k / 2 - 1 < nums1.size) nums1[i + k / 2 - 1] else Int.MAX_VALUE
//    val midVal2 = if (j + k / 2 - 1 < nums2.size) nums2[j + k / 2 - 1] else Int.MAX_VALUE
//    return if (midVal1 < midVal2) {
//      findKth(nums1, i + k / 2, nums2, j, k - k / 2)
//    } else {
//      findKth(nums1, i, nums2, j + k / 2, k - k / 2)
//    }
//  }

fun letterCombinations(digits: String): List<String> {
  if (digits.isEmpty()) return arrayListOf()
  val c = digits.first()
  val firstChars = getChers(c)
  val lastStr = letterCombinations(digits.removeRange(0,1))
  return if (lastStr.isEmpty()){
    firstChars
  } else {
    val res = arrayListOf<String>()
    for(chr in firstChars){
      res.addAll(lastStr.map { chr + it })
    }
    res
  }
}

fun getChers(c: Char):List<String>{
  return when(c){
    '2' -> listOf("a", "b", "c")
    '3' -> listOf("d", "e", "f")
    '4' -> listOf("g", "h", "i")
    '5' -> listOf("j", "k", "l")
    '6' -> listOf("m", "n", "o")
    '7' -> listOf("p", "q", "r", "s")
    '8' -> listOf("t", "u", "v")
    else -> listOf("w", "x", "y", "z")
  }
}

/**
 * 无重复字符的最长子串
 */
fun lengthOfLongestSubstring2(s: String): Int {
  if (s.isEmpty()) return 0
  val map = hashMapOf<Char, Int>()
  var max = 1
  var start = 0
  for (index in s.indices){
    val c = s[index]
    if (map.containsKey(c) && map[c]!! >= start){
      start = map[c]!! + 1
      map[c] = index
    } else {
      max = Math.max(max, index - start + 1)
      map[c] = index
    }
  }
  return max
}

/**
 * 合并有序链表
 */
fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
  return when {
    list1 == null && list2 == null -> null
    list1 != null && list2 == null -> list1
    list1 == null && list2 != null -> list2
    else -> {
      if (list1!!.`val` < list2!!.`val`){
        list1.next = mergeTwoLists(list1.next, list2)
        list1
      } else {
        list2.next = mergeTwoLists(list1, list2.next)
        list2
      }
    }
  }
}

fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
  if (l1 == null && l2 == null) return null
  if (l1 == null && l2 != null) return l2
  if (l2 == null && l1 != null) return l1

  fun addNode(l1: ListNode?, l2: ListNode?, value: Int): ListNode? {
    if (l1 == null && l2 == null && value == 0) return null
    val sum = (l1?.`val` ?: 0) + (l2?.`val` ?: 0) + value
    val node = ListNode(sum % 10)
    node.next = addNode(l1?.next, l2?.next, sum / 10)
    return node
  }

  return addNode(l1, l2, 0)
}

fun myPow(x: Double, n: Int): Double {
  when (n) {
    0 -> return 1.0
    1 -> return x
    -1 -> return 1 / x
  }
  val half = myPow(x, n / 2)
  return half * half * myPow(x, n % 2)
}

fun cuttingRope(n: Int): Int {
  when (n) {
    2 -> return 1
    3 -> return 2
    4 -> return 4
  }
  val last = n % 3
  var count = n / 3
  var lastSize = 1
  when (last) {
    1 -> {
      count--
      lastSize = 4
    }
    2 -> lastSize = 2
  }
  var result = 1
  for (i in 1..count) {
    result = (result * count) % 1000000007
  }

  return (result * lastSize) % 1000000007
}

fun movingCount(m: Int, n: Int, k: Int): Int {
  fun getNumSum(num: Int): Int {
    var sum = 0
    var temp = num
    while (temp >= 10) {
      sum += temp % 10
      temp /= 10
    }
    sum += temp
    return sum
  }

  var count = 0
  fun visit(i: Int, j: Int, target: Int, visited: Array<BooleanArray>) {
    if (i in 0 until m && j in 0 until n && getNumSum(i) + getNumSum(j) <= target && !visited[i][j]) {
      visited[i][j] = true
      count++
      visit(i - 1, j, target, visited)
      visit(i + 1, j, target, visited)
      visit(i, j - 1, target, visited)
      visit(i, j + 1, target, visited)
    }
  }

  val visited = Array(m) { BooleanArray(n) { false } }
  visit(0, 0, k, visited)
  return count
}

fun exist(board: Array<CharArray>, word: String): Boolean {
  if (board.isEmpty() || board[0].isEmpty()) return false

  fun search(i: Int, j: Int, targetPos: Int, visited: Array<BooleanArray>): Boolean {
    return when {
      i < 0 || i >= board.size || j < 0 || j >= board[0].size -> false
      visited[i][j] -> false
      board[i][j] != word[targetPos] -> false
      targetPos == word.length - 1 -> true
      else -> {
        visited[i][j] = true
        val result = search(i, j - 1, targetPos + 1, visited)
           || search(i - 1, j, targetPos + 1, visited)
           || search(i, j + 1, targetPos + 1, visited)
           || search(i + 1, j, targetPos + 1, visited)
        visited[i][j] = false
        result
      }
    }
  }

  val visited = Array(board.size) { BooleanArray(board[0].size) { false } }
  for (i in board.indices) {
    for (j in board[0].indices) {
      if (search(i, j, 0, visited)) {
        return true
      }
    }
  }
  return false
}

fun isStraight(nums: IntArray): Boolean {
  nums.sort()
  var zeroCount = 0
  var index = 0
  for (i in nums) {
    if (i != 0) {
      break
    }
    zeroCount++
    index++
  }

  for (i in index until nums.size - 1) {
    val next = nums[i + 1]
    if (nums[i] >= next) {
      return false
    }
    val gap = next - nums[i] - 1
    if (gap > zeroCount) {
      return false
    }
    zeroCount -= gap
  }
  return true
}

/**
 * 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部
 * 先局部翻转，再整体翻转
 */
fun reverseLeftWords(s: String, n: Int): String {
  fun swap(str: StringBuilder, i: Int, j: Int) {
    val temp = str[i]
    str[i] = str[j]
    str[j] = temp
  }

  fun reverseString(str: StringBuilder, start: Int, end: Int) {
    for (i in 0..(end - start) / 2) {
      swap(str, start + i, end - i)
    }
  }

  val str = java.lang.StringBuilder(s)
  reverseString(str, 0, n - 1)
  reverseString(str, n, s.length - 1)
  reverseString(str, 0, s.length - 1)
  return str.toString()
}

/**
 * 翻转单词顺序
 * 使用栈来实现
 */
fun reverseWords(s: String): String {
  var result = ""
  val stack = Stack<Char>()
  for (index in s.length - 1 downTo 0) {
    val c = s.get(index)
    if (c != ' ') {
      stack.push(c)
    } else {
      if (result.isNotEmpty()) {
        result += " "
      }
      while (stack.isNotEmpty()) {
        result += stack.pop()
      }
    }
  }
  if (stack.isNotEmpty()) {
    result += " "
  }
  while (stack.isNotEmpty()) {
    result += stack.pop()
  }
  return result
}

/**
 * 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
 * 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
 * 数学问题,假设有n个数的和刚好是target，通过等差数列可以算出首项是多少
 * n就从2开始遍历，最多有Math.pow((2.0 * target),0.5).toInt() + 1个数，复杂度在logn内
 * 算出首项小于1，说明n就太大了
 * 另一个解法是滑动窗口
 */
fun findContinuousSequence(target: Int): Array<IntArray> {
  val result = ArrayList<IntArray>()
  var num = 2
  while (true) {
    val first = (2 * target / num.toFloat() + 1 - num) / 2f
    if (first < 1) break
    if (!(first > first.toInt() || first < first.toInt())) {
      val array = IntArray(num)
      var temp = first.toInt()
      for (i in 0 until num) {
        array[i] = temp++
      }
      result.add(0, array)
    }
    num++
  }
  return result.toTypedArray()
}

fun twoSum(nums: IntArray, target: Int): IntArray {
  val result = intArrayOf(0, 0)
  var start = 0
  var end = nums.size - 1
  while (start < end) {
    if (nums[start] + nums[end] == target) {
      result[0] = nums[start]
      result[1] = nums[end]
      break
    }
    if (target - nums[end] > nums[start]) {
      start++
    } else end--
  }
  return result
}

/**
 * 判断二叉平衡树
 */
fun isBalanced(root: TreeNode?): Boolean {

  fun gettreeDepth(node: TreeNode?): Int {
    if (node == null) return 0
    val leftDepth = gettreeDepth(node.left)
    if (leftDepth == -1) return -1
    val rightDepth = gettreeDepth(node.right)
    if (rightDepth == -1) return -1
    if (kotlin.math.abs(leftDepth - rightDepth) > 1) {
      return -1
    }
    return kotlin.math.max(leftDepth, rightDepth) + 1
  }

  val depth = gettreeDepth(root)
  return depth != -1
}

/**
 * 二叉树的深度
 */
fun maxDepth2(root: TreeNode?): Int {
  if (root == null) return 0
  return kotlin.math.max(maxDepth(root.left), maxDepth(root.right)) + 1
}

/**
 * 二叉搜索数第k大的节点
 * 中序遍历
 */
fun kthLargest(root: TreeNode?, k: Int): Int {
  var count = 0
  fun searchNode(node: TreeNode?): Int? {
    if (node == null) return null
    val searchRight = searchNode(node?.right)
    if (searchRight != null) {
      return searchRight
    }
    count++
    if (count == k) return node.`val`
    return searchNode(node.left)
  }

  return searchNode(root) ?: 0
}

fun missingNumber(nums: IntArray): Int {
  val sum = (nums.size) * (nums.size + 1) / 2
  var aclSum = 0
  for (item in nums) {
    aclSum += item
  }
  return sum - aclSum
}

fun search(nums: IntArray, target: Int): Int {
  if (nums.isEmpty()) return 0
  if (target < nums.first() || target > nums.last()) return 0
  if (nums.size == 1 && nums.first() == target) return 1
  var start = 0
  var end = nums.size - 1
  var index = -1
  while (start < end) {
    if (end - start == 1) {
      if (nums[start] == target) {
        index = start
      } else if (nums[end] == target) {
        index = end
      }
      break
    }
    val middle = (end - start) / 2 + start
    if (nums[middle] > target) {
      end = middle
    } else if (nums[middle] == target) {
      index = middle
      break
    } else {
      start = middle
    }
  }

  if (index == -1) return 0

  var count = 1
  var i = index - 1
  while (i >= 0 && nums[i] == target) {
    count++
    i--
  }
  i = index + 1
  while (i < nums.size && nums[i] == target) {
    count++
    i++
  }
  return count
}


fun firstUniqChar(s: String): Char {
  val map = hashMapOf<Char, Int>()
  for (i in s.indices) {
    val c = s[i]
    if (map.containsKey(c)) {
      val num = map[c] as Int
      map[c] = num + 1
    } else {
      map[c] = 1
    }
  }

  for (i in s.indices) {
    val c = s[i]
    if (map[c] == 1) {
      return c
    }
  }
  return ' '
}

interface ITest {
  fun testfun()
}

fun inorderSuccessor(root: TreeNode?, p: TreeNode?): TreeNode? {
  if (root == null || p == null) return null
  return if (root.`val` < p.`val`) {
    inorderSuccessor(root.right, p)
  } else {
    inorderSuccessor(root.left, p) ?: root
  }
}

suspend fun testCorotines() {

  val handler = CoroutineExceptionHandler { _, t ->
    println(t.message)
  }

  GlobalScope.launch {
    GlobalScope.launch(handler) {
      launch {
        throw Exception("ssss")
      }
    }

    launch {

    }
  }
}


fun testKotlinThread() {
  var a = 0

  fun add() {
    synchronized(a) {
      a++
    }
  }

  val t1 = Thread {
    for (j in 0 until 60) {
      add()
    }
  }
  val t2 = Thread {
    for (j in 0 until 60) {
      add()
    }
  }
  t1.start()
  t2.start()
  while (Thread.activeCount() > 2) {
    Thread.yield()
  }
  print("i = $a")
}

fun reverseOnlyLetters(s: String): String {
  val str = s.toCharArray()
  var start = 0
  var end = s.length - 1
  while (start < end) {
    if (!str[start].isLetter()) {
      start++
      continue
    }
    if (!str[end].isLetter()) {
      end--
      continue
    }
    val temp = str[start]
    str[start] = str[end]
    str[end] = temp
    start++
    end--
  }
  return String(str)
}

fun maxDepth(root: TreeNode?): Int {
  if (root == null) return 0
  return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1
}

fun singleNumber(nums: IntArray): Int {
  var a = 0
  for (num in nums){
    a = a xor num
  }
  return a
}

fun climbStairs(n: Int): Int {
  if (n==1) return 1
  if (n==2) return 2
  var first = 1
  var second = 2
  for (i in 3..n){
    val temp = second
    second += first
    first = temp
  }
  return  second
}

/**
 * 最大连续子数组和
 * 这个算法已经考虑了全部为负数的可能性，在每一次累加之前先把sum为负数改成0，
 * 之后都会记录最大和，即使全部是负数也能记录下来
 */
fun maxSubArray(nums: IntArray): Int {
  var max = nums.firstOrNull() ?: Int.MIN_VALUE
  var sum = 0
  for (num in nums) {
    if (sum < 0) {
      sum = 0
    }
    sum += num
    max = Math.max(max, sum)
  }
  return max
}

class Singleton {
  companion object {
    fun getInstance(): Singleton = Holder.instance
  }

  private object Holder {
    val instance = Singleton()
  }
}

fun getLeastNumbers(arr: IntArray, k: Int): IntArray {
  val minHeap = PriorityQueue<Int>()
  for (item in arr) {
    minHeap.add(item)
  }
  val result = arrayListOf<Int>()
  for (i in minHeap.indices) {
    if (i == k) break
    result.add(minHeap.poll())
  }
  return result.toIntArray()
}


fun majorityElement(nums: IntArray): Int {
  var num = nums[0]
  var time = 0
  for (item in nums) {
    if (time == 0) {
      num = item
      time++
    } else {
      if (item == num) {
        time++
      } else {
        time--
      }
    }
  }
  return num
}

data class Node(
  val node: TreeNode,
  val line: Int = 0
)

fun levelOrder(root: TreeNode?): List<List<Int>> {
  if (root == null) return arrayListOf(arrayListOf())
  val queue = LinkedList<Node>()
  var currentLine = 0
  queue.add(Node(root, 0))
  val result = arrayListOf(arrayListOf<Int>())
  while (queue.isNotEmpty()) {
    val node = queue.pollFirst()
    if (node.line > currentLine) {
      currentLine++
      result.add(arrayListOf<Int>())
    }
    if (node.node.left != null) {
      queue.addLast(Node(node.node.left!!, node.line + 1))
    }
    if (node.node.right != null) {
      queue.addLast(Node(node.node.right!!, node.line + 1))
    }
    result.last().add(node.node.`val`)
  }
  return result
}

fun spiralOrder(matrix: Array<IntArray>): IntArray {
  val row = matrix.size
  val col = matrix.firstOrNull()?.size ?: return intArrayOf()
  var r = 0
  var c = 0
  var dir = 0
  var top = 0
  var left = 0
  var right = col - 1
  var bottom = row - 1
  val result = arrayListOf<Int>()
  for (i in 0 until col * row) {
    result.add(matrix[r][c])
    when (dir) {
      0 -> {
        if (c == right) {
          r++
          top++
          dir = 1
        } else {
          c++
        }
      }
      1 -> {
        if (r == bottom) {
          c--
          right--
          dir = 2
        } else {
          r++
        }
      }
      2 -> {
        if (c == left) {
          r--
          bottom--
          dir = 3
        } else {
          c--
        }
      }
      3 -> {
        if (r == top) {
          c++
          left++
          dir = 0
        } else {
          r--
        }
      }
    }
  }
  return result.toIntArray()
}


class MinStack() {

  /** initialize your data structure here. */
  val stack = Stack<Int>()
  val minStack = Stack<Int>()

  fun push(x: Int) {
    stack.push(x)
    val min = min() ?: Int.MAX_VALUE
    if (x < min) {
      minStack.push(x)
    } else {
      minStack.push(min)
    }
  }

  fun pop() {
    stack.pop()
    minStack.pop()
  }

  fun top(): Int {
    return stack.peek()
  }

  fun min(): Int? {
    if (minStack.isEmpty()) return null
    return minStack.peek()
  }

}

fun countKDifference(nums: IntArray, k: Int): Int {
  val hashMap = hashMapOf<Int, LinkedList<Int>>()
  var count = 0
  for (i in nums.indices) {
    val index = nums.size - 1 - i

    val target1 = nums[index] - k
    val target2 = nums[index] + k

    if (hashMap.containsKey(target1)) {
      count += hashMap[target1]!!.size
    }
    if (hashMap.containsKey(target2)) {
      count += hashMap[target2]!!.size
    }

    if (hashMap.containsKey(nums[index])) {
      val list = hashMap[nums[index]]!!
      list.addFirst(index)
    } else {
      hashMap[nums[index]] = LinkedList()
      hashMap[nums[index]]!!.addFirst(index)
    }
  }
  return count
}

class StockPrice() {
  data class Stock(
    var price: Int,
    var time: Int
  )

  private val list = arrayListOf<Stock>()
  private val map = hashMapOf<Int, Int>() // time to index
  private var maxTime = Int.MIN_VALUE

  private var maxPrice = Int.MIN_VALUE
  private var minPrice = Int.MAX_VALUE

  fun update(timestamp: Int, price: Int) {
    if (map.containsKey(timestamp)) {
      val index = map[timestamp]!!
      list.getOrNull(index)?.price = price
      maxPrice = Int.MIN_VALUE
      minPrice = Int.MAX_VALUE
      for (item in list) {
        if (maxPrice < item.price) {
          maxPrice = item.price
        }
        if (minPrice > item.price) {
          minPrice = item.price
        }
      }
    } else {
      list.add(Stock(price, timestamp))
      map[timestamp] = list.size - 1
      if (maxPrice < price) {
        maxPrice = price
      }
      if (minPrice > price) {
        minPrice = price
      }
    }

    if (timestamp > maxTime) {
      maxTime = timestamp
    }
  }

  fun current(): Int {
    return list[map[maxTime]!!].price
  }

  fun maximum(): Int {
    return maxPrice
  }

  fun minimum(): Int {
    return minPrice
  }
}


fun removePalindromeSub(s: String): Int {
  if (s.isEmpty() || s.length == 1) return s.length
  var start = 0
  var end = s.length - 1
  val isOdd = s.length % 2 == 1
  while (true) {
    if (start == end && isOdd) break
    if (end - start == 1 && !isOdd) break
    if (s[start] != s[end]) break
    start++
    end--
  }
  if ((isOdd && start == end) || (!isOdd && end - start == 1)) return 1
  return 2
}

fun minJumps(arr: IntArray): Int {
  val indexMap = hashMapOf<Int, ArrayList<Int>>()
  arr.forEachIndexed { index, item ->
    if (indexMap.containsKey(item)) {
      indexMap[item]!!.add(index)
    } else {
      indexMap[item] = arrayListOf(index)
    }
  }
  val stepArray = IntArray(arr.size) { Int.MAX_VALUE }
  stepArray[arr.size - 1] = 0

  // 设置函数
  fun setStep(index: Int) {
    val currentStep = stepArray[index]
    if (index < arr.size - 1) {
      if (currentStep + 1 < stepArray[index + 1]) {
        stepArray[index + 1] = currentStep + 1
        setStep(index + 1)
      }
    }
    if (index > 0) {
      if (currentStep + 1 < stepArray[index - 1]) {
        stepArray[index - 1] = currentStep + 1
        setStep(index - 1)
      }
    }
    val sameValueArray = indexMap[arr[index]] ?: return
    for (i in sameValueArray) {
      if (i == index) continue
      if (i < arr.size - 1) {
        if (currentStep + 1 < stepArray[i]) {
          val hasVisited = stepArray[i] != Int.MAX_VALUE
          stepArray[i] = currentStep + 1
          if (!hasVisited) {
            setStep(i)
          }
        }
      }
      if (i > 0) {
        if (currentStep + 1 < stepArray[i]) {
          val hasVisited = stepArray[i] != Int.MAX_VALUE
          stepArray[i] = currentStep + 1
          if (!hasVisited) {
            setStep(i)
          }
        }
      }
    }
  }

  setStep(arr.size - 1)
  return stepArray[0]
}

fun mirrorTree(root: TreeNode?): TreeNode? {
  if (root == null) return null
  val temp = root.left
  root.left = root.right
  root.right = temp
  mirrorTree(root.left)
  mirrorTree(root.right)
  return root
}

fun getKthFromEnd(head: ListNode?, k: Int): ListNode? {
  var first = head
  for (i in 0 until k) {
    if (first == null) return null
    first = first.next
  }
  var second = head
  while (first != null) {
    first = first.next
    second = second?.next
  }
  return second
}

fun reverseList(head: ListNode?): ListNode? {
  if (head == null) return null
  var a = head
  var b: ListNode? = a.next ?: return a
  a.next = null
  while (b != null) {
    val temp = b.next
    b.next = a
    a = b
    b = temp
  }
  return a
}

fun containsNearbyDuplicate(nums: IntArray, k: Int): Boolean {
  val map = hashMapOf<Int, Int>()
  for (i in nums.indices) {
    if (!map.containsKey(nums[i])) {
      map[nums[i]] = i
    } else {
      if (i - map.get(nums[i])!! <= k) {
        return true
      } else {
        map[nums[i]] = i
      }
    }
  }
  return false
}


fun isValid(s: String): Boolean {
  val stack = Stack<Char>()
  for (i in s.indices) {
    val c = s[i]
    if (stack.isEmpty()) {
      stack.push(c)
    } else {
      val top = stack.peek()
      if ((top == '(' && c == ')')
        || (top == '[' && c == ']')
        || (top == '{' && c == '}')
      ) {
        stack.pop()
      } else {
        stack.push(c)
      }
    }
  }
  return stack.isEmpty()
}

fun findMinDifference(timePoints: List<String>): Int {
  if (timePoints.size <= 1) return 0
  val tempArrayList = arrayListOf<Int>()
  timePoints.forEach {
    val strings = it.split(":")
    val time = strings[0].toInt() * 60 + strings[1].toInt()
    tempArrayList.add(time)
    tempArrayList.add(time + 1440)
  }
  val sortedList = tempArrayList.sorted()
  var last = sortedList[0]
  var gap = Int.MAX_VALUE
  for (i in 1 until sortedList.size) {
    if (Math.abs(sortedList[i] - last) < gap) {
      gap = Math.abs(sortedList[i])
    }
    last = sortedList[i]
  }
  return gap
}

class ListNode(var `val`: Int) {
  public var next: ListNode? = null
}

fun minArray(numbers: IntArray): Int {
  var max = numbers[0]
  for (item in numbers) {
    if (item >= max) {
      max = item
    } else {
      return item
    }
  }
  return numbers[0]
}

fun countVowelPermutation(n: Int): Int {
  if (n == 1) return 5
  val m = (Math.pow(10.0, 9.0) + 7).toLong()
  var size = LongArray(5) { 1 }
  for (i in 1 until n) {
    val tempSize = LongArray(5) { 1 }
    tempSize[0] = (size[1] + size[2] + size[4]) % m
    tempSize[1] = (size[0] + size[2]) % m
    tempSize[2] = (size[1] + size[3]) % m
    tempSize[3] = size[2] % m
    tempSize[4] = (size[2] + size[3]) % m
    size = tempSize
  }
  return ((size[0] + size[1] + size[2] + size[3] + size[4]) % m).toInt()
}


fun totalMoney(n: Int): Int {
  val week = n / 7
  val day = n % 7
  return (28 + 28 + (week - 1) * 7) * week / 2 + (1 + day) * day / 2 + day * week
}

fun kSmallestPairs(nums1: IntArray, nums2: IntArray, k: Int): List<List<Int>> {
  if (nums1.isEmpty() || nums2.isEmpty()) return arrayListOf()
  val result = arrayListOf<List<Int>>()
  val step = IntArray(nums1.size) { 0 }
  var time = 0
  while (time < nums1.size * nums2.size) {
    if (result.size == k) break
    val aStepIndex = findMin(step, nums1, nums2)
    result.add(arrayListOf(nums1[aStepIndex], nums2[step[aStepIndex]]))
    step[aStepIndex]++
    time++
  }
  return result
}

fun numWays(n: Int): Int {
  when (n) {
    0 -> return 1
    1, 2 -> return n
  }
  var first = 1
  var second = 2
  for (i in 3..n) {
    second = (first + second)
    first = second - first
    second %= 1000000007
  }
  return second
}

fun findMin(step: IntArray, a: IntArray, b: IntArray): Int {
  var min = Int.MAX_VALUE
  var targetIndex = 0
  for (index in a.indices) {
    if (step[index] >= b.size) continue
    val value = a[index] + b[step[index]]
    if (value < min) {
      min = value
      targetIndex = index
    }
  }
  return targetIndex
}


class TreeNode(var `val`: Int) {
  var left: TreeNode? = null
  var right: TreeNode? = null
}

fun buildTree(preorder: IntArray, inorder: IntArray): TreeNode? {
  if (preorder.isEmpty()) return null
  if (preorder.size == 1) return TreeNode(preorder.first())
  if (preorder.size == 2) return TreeNode(preorder.first()).apply {
    if (inorder.first() == `val`) {
      right = TreeNode(inorder[1])
    } else {
      left = TreeNode(inorder[0])
    }
  }

  val node = TreeNode(preorder[0])
  val indexInInOrder = inorder.indexOfFirst { it == node.`val` }
  val leftInOrder = inorder.filterIndexed { index, i -> index < indexInInOrder }.toIntArray()
  val rightInOrder = inorder.filterIndexed { index, i -> index > indexInInOrder }.toIntArray()
  val indecInPreOrder = leftInOrder.size
  val leftPreOrder = preorder.filterIndexed { index, i -> index <= indecInPreOrder && index != 0 }.toIntArray()
  val rightPreOrder = preorder.filterIndexed { index, i -> index > indecInPreOrder && index != 0 }.toIntArray()
  return node.apply {
    left = buildTree(leftPreOrder, leftInOrder)
    right = buildTree(rightPreOrder, rightInOrder)
  }
}

fun printNode(node: TreeNode?) {
  if (node == null) {
    print("null,")
  } else {
    print("${node.`val`},")
    printNode(node.left)
    printNode(node.right)
  }
}


fun fib(n: Int): Int {
  return when (n) {
    0, 1 -> n
    else -> {
      var first = 0
      var second = 1
      for (index in 2..n) {
        second = (first + second)
        first = second - first
        second %= 1000000007
      }
      second
    }
  }
}

fun dominantIndex(nums: IntArray): Int {
  if (nums.size < 2) return -1
  var max = Int.MIN_VALUE
  var secondMax = Int.MIN_VALUE
  var maxIndex = 0
  for (num in nums.withIndex()) {
    if (num.value > max) {
      secondMax = max
      max = num.value
      maxIndex = num.index
    } else if (num.value > secondMax) {
      secondMax = num.value
    }
  }
  return if (max >= secondMax * 2) maxIndex else -1
}

fun increasingTriplet(nums: IntArray): Boolean {
  if (nums.size < 3) return false
  var a = Int.MAX_VALUE
  var b = Int.MAX_VALUE
  for (index in nums.indices) {
    val n = nums[index]
    when {
      n < a -> a = n
      n < b -> b = n
      else -> return true
    }
  }
  return false
}

class CQueue {

  val q = PriorityQueue<Int>()
  fun appendTail(value: Int) {
    q.add(value)
  }

  fun deleteHead(): Int {
    if (q.isEmpty()) return -1
    return q.poll()
  }
}

fun isAdditiveNumber(num: String): Boolean {
  if (num.length < 3) return false
  val totalLength = num.length
  var result = false
//    val maxSize = Math.ceil(totalLength.toDouble() / 2f).toInt()
  val maxSize = totalLength / 2
  out@ for (firstSize in 1..maxSize) {
    for (secondSize in 1..maxSize) {
      if (firstSize > totalLength - firstSize - secondSize || secondSize > totalLength - firstSize - secondSize) break
      for (thirdSize in 1..maxSize) {
        if (firstSize + secondSize + thirdSize > num.length) continue
        val firstWord = num.subSequence(0, firstSize).toString().trim()
        if (firstSize > 1 && firstWord.startsWith("0")) continue
        val firstNum = firstWord.toLong()

        val secondWord = num.subSequence(firstSize, firstSize + secondSize).toString().trim()
        if (secondSize > 1 && secondWord.startsWith("0")) continue
        val secondNum = secondWord.toLong()

        val thirdWord =
          num.subSequence(firstSize + secondSize, thirdSize + firstSize + secondSize).toString().trim()
        if (thirdSize > 1 && thirdWord.startsWith("0")) continue
        val thirdNum = thirdWord.toLong()

        val subString = num.subSequence(thirdSize + firstSize + secondSize, num.length).toString().trim()
        if (firstNum + secondNum == thirdNum && (hasThirdNumber(
            secondWord,
            thirdWord,
            subString
          ) || subString.isEmpty())
        ) {
          result = true
          break@out
        }
      }
    }
  }
  return result
}

fun hasThirdNumber(firstStr: String, secondStr: String, num: String): Boolean {
  if (num.length < firstStr.length || num.length < secondStr.length) return false
  if (num.startsWith("0")) return false
  for (thirdSize in 1..num.length) {
    val thirdStr = num.subSequence(0, thirdSize).toString().trim()
    val thirdNum = thirdStr.toLong()
    val lastStr = num.subSequence(thirdSize, num.length).toString().trim()
    if (firstStr.toLong() + secondStr.toLong() == thirdNum && (hasThirdNumber(
        secondStr,
        thirdStr,
        lastStr
      ) || lastStr.isEmpty())
    ) {
      return true
    }
  }
  return false
}

