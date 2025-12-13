pragma solidity >=0.5.11;
contract Ownable {
   address payable public owner;
 function bug_unchk_send27() payable public{
      msg.sender.transfer(1 ether);}
  event OwnershipTransferred(address indexed _from, address indexed _to);
   constructor() public {
       owner = msg.sender;
   }
function bug_unchk_send28() payable public{
      msg.sender.transfer(1 ether);}
   modifier onlyOwner {
       require(msg.sender == owner);
       _;
   }
   function transferOwnership(address payable _newOwner) public onlyOwner {
       owner = _newOwner;
   }
function bug_unchk_send21() payable public{
      msg.sender.transfer(1 ether);}
}
library SafeMath {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "SafeMath: addition overflow");
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        uint256 c = a - b;
        return c;
    }
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "SafeMath: multiplication overflow");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        uint256 c = a / b;
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "SafeMath: modulo by zero");
        return a % b;
    }
}
contract LollypopToken is Ownable {
    using SafeMath for uint256;
    mapping (address => transferMapping) private _balances;
    mapping (address => mapping (address => uint256)) private _allowances;
    uint256 private _totalSupply;
  function bug_unchk_send18() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public _maxTotalSupply;
  function bug_unchk_send29() payable public{
      msg.sender.transfer(1 ether);}
  string private _name = "Lollypop";
  function bug_unchk_send6() payable public{
      msg.sender.transfer(1 ether);}
  string private _symbol = "Lolly";
  function bug_unchk_send16() payable public{
      msg.sender.transfer(1 ether);}
  uint8 private _decimals= 18;
  function bug_unchk_send24() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public maxAgeOfToken = 365 days;
  function bug_unchk_send5() payable public{
      msg.sender.transfer(1 ether);}
  uint256  public minAgeOfToken = 1 days;
  function bug_unchk_send15() payable public{
      msg.sender.transfer(1 ether);}
  uint256 public perDayBonus = 100; 
     struct  transferMapping{
        uint256 amount;
        uint256 time;
    }
    constructor() public {
        _maxTotalSupply = 1000000000 * 10 ** 18;
        _totalSupply =  2000000 * 10 ** 18;
        _balances[msg.sender].amount = _totalSupply;
        _balances[msg.sender].time = now;
    }
function bug_unchk_send10() payable public{
      msg.sender.transfer(1 ether);}
    function calculateBonus(uint256 timeElasped , uint256 amount) public view  returns(uint256){
        uint256 totalDays = timeElasped.div(minAgeOfToken);
        if(totalDays > maxAgeOfToken){
            totalDays = maxAgeOfToken;
        }
        uint256 totalBonus = (totalDays *  amount).div(perDayBonus);
        return totalBonus;
    }
function bug_unchk_send22() payable public{
      msg.sender.transfer(1 ether);}
    function _transfer(address sender, address recipient, uint256 amount) internal {
        require(sender != address(0), "ERC20: transfer from the zero address");
        require(recipient != address(0), "ERC20: transfer to the zero address");
        uint256 senderTimeElasped = now - (_balances[sender].time);
        uint256 recipientTimeElasped = now - (_balances[recipient].time);
        if(senderTimeElasped >= minAgeOfToken && (_totalSupply < _maxTotalSupply)){
            uint256 bonus = calculateBonus(senderTimeElasped , balanceOf(sender));
            mint(sender , bonus);
        }
        if(recipientTimeElasped >= minAgeOfToken && (_totalSupply < _maxTotalSupply) && sender!= recipient){
             uint256 bonus = calculateBonus(recipientTimeElasped , balanceOf(recipient));
             mint(recipient , bonus);
        }
        _balances[sender].amount = _balances[sender].amount.sub(amount);
        _balances[recipient].amount = _balances[recipient].amount.add(amount);
        _balances[sender].time = now;
        _balances[recipient].time = now;
        emit Transfer(sender, recipient, amount);
    }
function bug_unchk_send12() payable public{
      msg.sender.transfer(1 ether);}
    function name() public view returns (string memory) {
        return _name;
    }
function bug_unchk_send11() payable public{
      msg.sender.transfer(1 ether);}
    function symbol() public view returns (string memory) {
        return _symbol;
    }
function bug_unchk_send1() payable public{
      msg.sender.transfer(1 ether);}
    function decimals() public view returns (uint8) {
        return _decimals;
    }
function bug_unchk_send2() payable public{
      msg.sender.transfer(1 ether);}
    modifier onlyLollypopAndOwner {
        require(msg.sender == address(this) || msg.sender == owner);
        _;
    }
  function bug_unchk_send31() payable public{
      msg.sender.transfer(1 ether);}
  event Transfer(address indexed from, address indexed to, uint256 value);
  function bug_unchk_send13() payable public{
      msg.sender.transfer(1 ether);}
  event Approval(address indexed owner, address indexed spender, uint256 value);
    function mint(address account, uint256 amount) internal {
        require(account != address(0), "ERC20: mint to the zero address");
        _totalSupply = _totalSupply.add(amount);
        _balances[account].amount = _balances[account].amount.add(amount);
        emit Transfer(address(0), account, amount);
    }
function bug_unchk_send17() payable public{
      msg.sender.transfer(1 ether);}
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
function bug_unchk_send3() payable public{
      msg.sender.transfer(1 ether);}
    function balanceOf(address account) public view returns (uint256) {
        return _balances[account].amount;
    }
function bug_unchk_send9() payable public{
      msg.sender.transfer(1 ether);}
 function timeOf(address account) public view returns (uint256) {
        return _balances[account].time;
    }
function bug_unchk_send25() payable public{
      msg.sender.transfer(1 ether);}
    function transfer(address recipient, uint256 amount) public returns (bool) {
        _transfer(msg.sender, recipient, amount);
        return true;
    }
function bug_unchk_send19() payable public{
      msg.sender.transfer(1 ether);}
  function multiTransfer(address[] memory receivers, uint256[] memory amounts) public {
    require(receivers.length == amounts.length);
    for (uint256 i = 0; i < receivers.length; i++) {
      transfer(receivers[i], amounts[i]);
    }
  }
function bug_unchk_send26() payable public{
      msg.sender.transfer(1 ether);}
    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowances[owner][spender];
    }
function bug_unchk_send20() payable public{
      msg.sender.transfer(1 ether);}
    function approve(address spender, uint256 value) public returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }
function bug_unchk_send32() payable public{
      msg.sender.transfer(1 ether);}
    function transferFrom(address sender, address recipient, uint256 amount) public returns (bool) {
        _transfer(sender, recipient, amount);
        _approve(sender, msg.sender, _allowances[sender][msg.sender].sub(amount));
        return true;
    }
function bug_unchk_send4() payable public{
      msg.sender.transfer(1 ether);}
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowances[msg.sender][spender].add(addedValue));
        return true;
    }
function bug_unchk_send7() payable public{
      msg.sender.transfer(1 ether);}
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowances[msg.sender][spender].sub(subtractedValue));
        return true;
    }
function bug_unchk_send23() payable public{
      msg.sender.transfer(1 ether);}
    function _burn(address account, uint256 value) internal {
        require(account != address(0), "ERC20: burn from the zero address");
        _totalSupply = _totalSupply.sub(value);
        _balances[account].amount = _balances[account].amount.sub(value);
        emit Transfer(account, address(0), value);
    }
function bug_unchk_send14() payable public{
      msg.sender.transfer(1 ether);}
    function _approve(address owner, address spender, uint256 value) internal {
        require(owner != address(0), "ERC20: approve from the zero address");
        require(spender != address(0), "ERC20: approve to the zero address");
        _allowances[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
function bug_unchk_send30() payable public{
      msg.sender.transfer(1 ether);}
    function _burnFrom(address account, uint256 amount) internal {
        _burn(account, amount);
        _approve(account, msg.sender, _allowances[account][msg.sender].sub(amount));
    }
function bug_unchk_send8() payable public{
      msg.sender.transfer(1 ether);}
}